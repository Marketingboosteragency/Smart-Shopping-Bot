# app.py (versión con búsqueda avanzada por IA)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 5.0 (AI-Powered Search)
# Novedades:
# - Integración de lógica de búsqueda avanzada con scraping y verificación por IA.
# - Uso de búsqueda orgánica de Google como fuente principal.
# - Análisis estadístico de precios para descartar resultados ilógicos.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
import os
import time
from collections import Counter
import json
import statistics  # GÉNESIS: Necesario para el análisis de precios
from bs4 import BeautifulSoup # GÉNESIS: Necesario para el scraping

# --- IMPORTS DE FLASK Y SEGURIDAD ---
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash

# --- IMPORTACIÓN DE GOOGLE CLOUD VISION ---
try:
    from google.cloud import vision
    print("✅ Módulo de Google Cloud Vision importado.")
except ImportError:
    print("❌ ERROR: El módulo 'google-cloud-vision' no está instalado.")
    vision = None

# --- GÉNESIS: IMPORTACIÓN DE GOOGLE GENERATIVE AI (GEMINI) ---
try:
    import google.generativeai as genai
    print("✅ Módulo de Google Generative AI (Gemini) importado.")
except ImportError:
    print("⚠️ AVISO: El módulo 'google-generativeai' no está instalado. La verificación por IA no funcionará.")
    genai = None

# =============== CONFIGURACIÓN DE APIs MEDIANTE VARIABLES DE ENTORNO ===============

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get('GOOGLE_CREDENTIALS_JSON')
# GÉNESIS: Nueva clave para la API de Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Configuración de Gemini si la clave está presente
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR: No se pudo configurar la API de Gemini: {e}")
        genai = None

# --- INICIALIZACIÓN DE GOOGLE CLOUD VISION ---
if GOOGLE_CREDENTIALS_JSON_STR and vision:
    try:
        google_creds_info = json.loads(GOOGLE_CREDENTIALS_JSON_STR)
        temp_creds_path = '/tmp/google-credentials.json'
        with open(temp_creds_path, 'w') as f:
            json.dump(google_creds_info, f)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_path
        print("✅ Credenciales de Google Vision cargadas desde variable de entorno.")
    except Exception as e:
        print(f"❌ ERROR: No se pudieron cargar las credenciales de Google Vision: {e}")

# ==============================================================================
# SECCIÓN 1: LÓGICA DEL SMART SHOPPING BOT (MODIFICADA)
# ==============================================================================

# GÉNESIS: FUNCIONES AUXILIARES PARA LA BÚSQUEDA AVANZADA
def _deep_scrape_content(url: str) -> Dict:
    """
    PLACEHOLDER: Realiza un scraping profundo de una URL.
    DEBES REEMPLAZAR ESTO con tu lógica de scraping robusta.
    """
    headers = {'User-Agent': UserAgent().random}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        price_text = "N/A"
        # Lógica de ejemplo muy básica para encontrar el precio
        price_tags = soup.find_all(text=re.compile(r'\$\s?\d+([.,]\d+)?'))
        if price_tags:
            match = re.search(r'(\d+([.,]\d+)?)', price_tags[0])
            if match:
                price_text = match.group(0).replace(',', '')
        title = soup.title.string if soup.title else 'Sin título'
        text_content = ' '.join(soup.stripped_strings)[:1000]
        print(f"    Scraping en {url[:40]}... Título: '{title[:30]}...', Precio: {price_text}")
        return {'title': title, 'text': text_content, 'price': price_text}
    except Exception as e:
        print(f"    Falló el scraping en {url[:40]}: {e}")
        return {'title': 'N/A', 'text': '', 'price': 'N/A'}

def _verify_product_with_gemini(query: str, product_title: str, product_text: str) -> bool:
    """
    PLACEHOLDER: Usa Gemini para verificar si el contenido es relevante.
    """
    if not genai:
        print("    (Saltando verificación IA: Gemini no configurado)")
        return True
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Is the following product relevant to the user's search? Answer only with 'SI' or 'NO'.
        User search: "{query}"
        Page title: "{product_title}"
        Page text extract: "{product_text[:500]}"
        """
        response = model.generate_content(prompt)
        decision = response.text.strip().upper()
        print(f"    Verificación IA para '{product_title[:30]}...': {decision}")
        return "SI" in decision
    except Exception as e:
        print(f"    Error en verificación IA: {e}")
        return False

def _get_clean_company_name(item: Dict) -> str:
    """
    PLACEHOLDER: Extrae un nombre de tienda limpio.
    """
    try:
        if source := item.get('source'): return source
        return urlparse(item.get('link', '')).netloc.replace('www.', '').split('.')[0].capitalize()
    except:
        return "Tienda"

# --- CLASES PRINCIPALES ---
@dataclass
class ProductResult:
    name: str; price: float; store: str; url: str; image_url: str = ""; rating: float = 0.0; reviews: int = 0; availability: str = "In Stock"; shipping: str = ""; original_price: float = 0.0; discount: str = ""; seller: str = ""

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.ua = UserAgent()
        if vision and GOOGLE_CREDENTIALS_JSON_STR:
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                print("✅ Cliente de Google Cloud Vision inicializado.")
            except Exception as e:
                print(f"❌ ERROR CRÍTICO EN VISION INIT: {e}")
                self.vision_client = None
        else:
            self.vision_client = None

    def get_query_from_image_vision_api(self, image_content: bytes) -> Optional[str]:
        if not self.vision_client: return None
        print("  🧠 Analizando imagen con Google Cloud Vision API...")
        try:
            image = vision.Image(content=image_content)
            response = self.vision_client.web_detection(image=image)
            if response.web_detection and response.web_detection.best_guess_labels:
                return response.web_detection.best_guess_labels[0].label
            return None
        except Exception as e:
            print(f"  ❌ Fallo en análisis con Google Cloud Vision: {e}")
            return None

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], bool]:
        print("\n🚀 Iniciando Smart Shopping Bot...")
        final_query = query
        if image_content:
            final_query = self.get_query_from_image_vision_api(image_content)
        
        if not final_query:
            print("❌ No se pudo determinar una consulta válida.")
            return [], False
        
        print(f"🔍 Lanzando búsqueda AVANZADA para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query)
        return best_deals, False

    def search_with_ai_verification(self, search_query: str) -> List[ProductResult]:
        """ GÉNESIS: Esta es la nueva función de búsqueda avanzada integrada. """
        print(f"--- Iniciando búsqueda de PRODUCTOS para: '{search_query}' ---")
        params = {
            "q": search_query, "engine": "google", "location": "United States",
            "gl": "us", "hl": "en", "num": "50", "api_key": self.serpapi_key
        }
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            organic_results = response.json().get('organic_results', [])
            print(f"SerpApi encontró {len(organic_results)} resultados orgánicos iniciales.")
            
            results_with_price = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_item = {executor.submit(_deep_scrape_content, item.get('link')): item for item in organic_results if item.get('link')}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    content = future.result()
                    if content and content['price'] != "N/A":
                        if _verify_product_with_gemini(search_query, content['title'], content['text']):
                            store_name = _get_clean_company_name(item)
                            results_with_price.append({
                                'store': store_name, 'product_name': item.get('title', 'Sin título'),
                                'price_float': float(content['price'].replace(',', '')), 'url': item.get('link'),
                                'image_url': item.get('thumbnail', '')
                            })
            
            print(f"Se encontraron {len(results_with_price)} resultados RELEVANTES Y CON PRECIO después de la verificación IA.")
            
            if len(results_with_price) < 2:
                final_results_dict = results_with_price
            else:
                prices = [r['price_float'] for r in results_with_price]
                mean_price = statistics.mean(prices)
                price_threshold = max(0.50, mean_price / 10)
                print(f"Análisis de precios: Media=${mean_price:.2f}, Umbral Mínimo Lógico=${price_threshold:.2f}")
                final_results_dict = [r for r in results_with_price if r['price_float'] >= price_threshold]

            final_results_obj = [ProductResult(name=res['product_name'], price=res['price_float'], store=res['store'], url=res['url'], image_url=res.get('image_url', '')) for res in final_results_dict]
            final_results_obj.sort(key=lambda x: x.price)
            
            print(f"✅ Se encontraron {len(final_results_obj)} ofertas válidas finales.")
            return final_results_obj[:30]

        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda avanzada: {e}")
            return []

# ==============================================================================
# SECCIÓN 2: CONFIGURACIÓN DE FLASK Y RUTAS (Sin cambios)
# ==============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-de-respaldo')
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('main_app_page'))
    return render_template_string(AUTH_TEMPLATE_LOGIN_ONLY)

@app.route('/login', methods=['POST'])
def login():
    # ... (código sin cambios)
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
        if error_message in ['INVALID_PASSWORD', 'EMAIL_NOT_FOUND']: flash('Correo o contraseña incorrectos.', 'danger')
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
    query = request.form.get('query'); image_file = request.files.get('image_file'); image_content = None
    if image_file and image_file.filename != '': image_content = image_file.read()
    results, is_alternative = shopping_bot.search_product(query=query, image_content=image_content)
    results_dicts = [p.__dict__ for p in results]; return jsonify(results=results_dicts, is_alternative=is_alternative)

# ==============================================================================
# SECCIÓN 5: PLANTILLAS Y EJECUCIÓN
# ==============================================================================
AUTH_TEMPLATE_LOGIN_ONLY = """ (Tu plantilla de login aquí sin cambios) """
SEARCH_TEMPLATE = """ (Tu plantilla de búsqueda aquí sin cambios) """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
