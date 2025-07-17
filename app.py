# app.py (versión 17.0 - Prueba de Fuego de SerpApi - Completo)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 17.0 (SerpApi Testbed)
# Novedades:
# - Se elimina TODA la lógica de IA y análisis de imagen para aislar el problema.
# - La aplicación ahora solo realiza una búsqueda de texto simple en Google Shopping.
# - Se añade logging de diagnóstico para la respuesta de SerpApi.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL DE FLASK Y APIS
# ==============================================================================
app = Flask(__name__)

# Configuración desde variables de entorno
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-muy-fuerte')

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (SIMPLIFICADA AL MÁXIMO)
# ==============================================================================

@dataclass
class ProductResult:
    name: str; price: float; store: str; url: str; image_url: str = ""

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key

    def search_product(self, query: str) -> List[ProductResult]:
        if not query:
            print("❌ No se proporcionó una consulta de búsqueda.")
            return []

        print(f"🚀 Lanzando búsqueda en Google Shopping para: '{query}'")
        
        params = {
            "q": query,
            "engine": "google_shopping",
            "location": "United States",
            "gl": "us",
            "hl": "en",
            "num": "100",
            "api_key": self.serpapi_key
        }
        
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            shopping_results = data.get('shopping_results', [])

            if not shopping_results:
                print("⚠️ SerpApi devolvió una respuesta válida pero sin 'shopping_results'.")
                print(f"   Respuesta completa de SerpApi: {json.dumps(data, indent=2)}")

            products = []
            for item in shopping_results:
                if all(k in item for k in ['price', 'title', 'link', 'source']):
                    try:
                        price_str = item.get('extracted_price', item.get('price', ''))
                        price_float = float(re.sub(r'[^\d.]', '', str(price_str)))
                        if price_float > 0:
                             products.append(ProductResult(name=item['title'], price=price_float, store=item['source'], url=item['link'], image_url=item.get('thumbnail', '')))
                    except (ValueError, TypeError):
                        continue
            
            products.sort(key=lambda x: x.price)
            print(f"✅ Búsqueda finalizada. Se encontraron {len(products)} resultados válidos.")
            return products

        except requests.exceptions.HTTPError as e:
            print(f"❌ Ocurrió un error HTTP en la búsqueda: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"❌ Ocurrió un error general en la búsqueda: {e}")
            return []

# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('main_app_page'))
    return render_template_string(AUTH_TEMPLATE_LOGIN_ONLY)

@app.route('/login', methods=['POST'])
def login():
    if not FIREBASE_WEB_API_KEY: flash('El servicio de autenticación no está configurado.', 'danger'); return redirect(url_for('index'))
    email = request.form.get('email'); password = request.form.get('password')
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {'email': email, 'password': password, 'returnSecureToken': True}
    try:
        response = requests.post(rest_api_url, json=payload); response.raise_for_status()
        user_data = response.json()
        session['user_id'] = user_data['localId']; session['user_name'] = user_data.get('displayName', email); session['id_token'] = user_data['idToken']
        flash('¡Has iniciado sesión correctamente!', 'success'); return redirect(url_for('main_app_page'))
    except requests.exceptions.HTTPError as e:
        error_json = e.response.json().get('error', {}); error_message = error_json.get('message', 'ERROR_DESCONOCIDO')
        if error_message in ['INVALID_PASSWORD', 'EMAIL_NOT_FOUND', 'INVALID_LOGIN_CREDENTIALS']: flash('Correo o contraseña incorrectos.', 'danger')
        else: flash(f'Error al iniciar sesión: {error_message}', 'danger')
        return redirect(url_for('index'))
    except Exception as e: flash(f'Ocurrió un error inesperado: {e}', 'danger'); return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear(); flash('Has cerrado la sesión.', 'success'); return redirect(url_for('index'))

@app.route('/app')
def main_app_page():
    if 'user_id' not in session: flash('Debes iniciar sesión para acceder a esta página.', 'warning'); return redirect(url_for('index'))
    user_name = session.get('user_name', 'Usuario'); return render_template_string(SEARCH_TEMPLATE, user_name=user_name)

@app.route('/api/search', methods=['POST'])
def api_search():
    if 'user_id' not in session: return jsonify({'error': 'No autorizado'}), 401
    query = request.form.get('query')
    # No se procesa la imagen en esta versión de diagnóstico
    results = shopping_bot.search_product(query=query)
    results_dicts = [res.__dict__ for res in results]
    return jsonify(results=results_dicts, suggestions=[])

# ==============================================================================
# SECCIÓN 4: PLANTILLAS HTML Y EJECUCIÓN
# ==============================================================================
AUTH_TEMPLATE_LOGIN_ONLY = """(Pega aquí tu plantilla de Login)"""
SEARCH_TEMPLATE = """(Pega aquí tu plantilla de Búsqueda)"""
# ... (asegúrate de que tus plantillas HTML completas están aquí)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
