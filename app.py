# app.py (versión 16.0 - Motor Ultra-Fiable de Google Shopping)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 16.0 (Ultra-Reliable Shopping Engine)
# Novedades:
# - Se elimina por completo el scraping para máxima velocidad y fiabilidad.
# - El motor de búsqueda se basa 100% en la API de Google Shopping.
# - Garantiza resultados precisos, baratos y geo-localizados para cualquier producto.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
import io
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash
from PIL import Image

# --- IMPORTS DE APIS DE GOOGLE ---
try:
    import google.generativeai as genai
    print("✅ Módulo de Google Generative AI (Gemini) importado.")
except ImportError:
    print("⚠️ AVISO: 'google-generativeai' no está instalado.")
    genai = None

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL DE FLASK Y APIS
# ==============================================================================
app = Flask(__name__)

# Configuración desde variables de entorno
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-muy-fuerte')

# Configuración de Gemini
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR al configurar API de Gemini: {e}")
        genai = None

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (SIMPLIFICADA Y ROBUSTA)
# ==============================================================================

@dataclass
class ProductResult:
    name: str
    price: float
    store: str
    url: str
    image_url: str = ""

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key

    def get_descriptive_query_from_image(self, image_content: bytes) -> Optional[str]:
        if not genai:
            print("  ❌ Análisis con Gemini Vision saltado: Modelo no configurado.")
            return None
        print("  🧠 Analizando imagen con Gemini Vision...")
        try:
            image_pil = Image.open(io.BytesIO(image_content))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = """You are an expert in identifying products. Analyze the image and generate a specific, effective search query in English to find this product for sale online. Respond ONLY with the search query."""
            response = model.generate_content([prompt, image_pil])
            query = response.text.strip().replace("*", "")
            print(f"  ✅ Consulta experta generada por Gemini Vision: '{query}'")
            return query
        except Exception as e:
            print(f"  ❌ Fallo CRÍTICO en análisis con Gemini Vision: {e}")
            return None

    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> List[ProductResult]:
        text_query = query.strip() if query else None
        image_query = self.get_descriptive_query_from_image(image_content) if image_content else None
        
        final_query = None
        if text_query and image_query:
            final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query:
            final_query = text_query
        elif image_query:
            final_query = image_query

        if not final_query:
            print("❌ No se pudo determinar una consulta válida.")
            return []

        print(f"🚀 Lanzando búsqueda en Google Shopping para: '{final_query}'")
        
        params = {
            "q": final_query,
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
            
            products = []
            for item in response.json().get('shopping_results', []):
                if all(k in item for k in ['price', 'title', 'link', 'source']):
                    try:
                        price_str = item.get('extracted_price', item['price'])
                        price_float = float(re.sub(r'[^\d.]', '', str(price_str)))
                        
                        if price_float >= 0.01:
                             products.append(ProductResult(
                                name=item['title'],
                                price=price_float,
                                store=item['source'],
                                url=item['link'],
                                image_url=item.get('thumbnail', '')
                            ))
                    except (ValueError, TypeError):
                        continue
            
            products.sort(key=lambda x: x.price)
            print(f"✅ Búsqueda finalizada. Se encontraron {len(products)} resultados válidos en Google Shopping.")
            return products

        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda de Google Shopping: {e}")
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
    image_file = request.files.get('image_file')
    image_content = image_file.read() if image_file and image_file.filename != '' else None
    results = shopping_bot.search_product(query=query, image_content=image_content)
    results_dicts = [res.__dict__ for res in results]
    return jsonify(results=results_dicts, suggestions=[])

# ==============================================================================
# SECCIÓN 4: PLANTILLAS HTML Y EJECUCIÓN
# ==============================================================================
AUTH_TEMPLATE_LOGIN_ONLY = """
<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Acceso | Smart Shopping Bot</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--text-color-dark:#2C3E50;--card-bg:#FFFFFF;--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:linear-gradient(135deg,var(--primary-color) 0%,var(--secondary-color) 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}.auth-container{max-width:480px;width:100%;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-medium);overflow:hidden;animation:fadeIn .8s ease-out}@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.form-header{text-align:center;padding:40px 30px 20px}.form-header h1{color:var(--text-color-dark);font-size:2em;margin-bottom:10px}.form-header p{color:#7f8c8d;font-size:1.1em}.form-body{padding:10px 40px 40px}form{display:flex;flex-direction:column;gap:20px}.input-group{display:flex;flex-direction:column;gap:8px}.input-group label{font-weight:600;color:var(--text-color-dark);font-size:.95em}.input-group input{padding:16px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:16px;transition:all .3s ease}.input-group input:focus{outline:0;border-color:var(--primary-color);box-shadow:0 0 0 4px rgba(74,144,226,.2)}.submit-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:16px 30px;font-size:1.1em;font-weight:600;border-radius:12px;cursor:pointer;transition:all .3s ease;margin-top:15px}.submit-btn:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(0,0,0,.2)}.flash-messages{list-style:none;padding:0 40px 20px}.flash{padding:15px;margin-bottom:15px;border-radius:8px;text-align:center}.flash.success{background-color:#d4edda;color:#155724}.flash.danger{background-color:#f8d7da;color:#721c24}.flash.warning{background-color:#fff3cd;color:#856404}</style></head><body><div class="auth-container"><div class="form-header"><h1>Bienvenido de Nuevo</h1><p>Accede para encontrar las mejores ofertas.</p></div>{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}<ul class=flash-messages>{% for category, message in messages %}<li class="flash {{ category }}">{{ message }}</li>{% endfor %}</ul>{% endif %}{% endwith %}<div class="form-body"><form id="login-form" action="{{ url_for('login') }}" method="post"><div class="input-group"><label for="login-email">Correo</label><input type="email" name="email" required></div><div class="input-group"><label for="login-password">Contraseña</label><input type="password" name="password" required></div><button type="submit" class="submit-btn">Entrar</button></form></div></div></body></html>
"""
SEARCH_TEMPLATE = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee;display:flex;flex-direction:column;position:relative}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center;overflow:hidden}.product-image img{width:100%;height:100%;object-fit:cover}.product-info{padding:25px;display:flex;flex-direction:column;flex-grow:1;justify-content:space-between}.product-title{font-size:1.1em;font-weight:600;margin-bottom:12px;color:var(--text-color-dark)}.price-store-wrapper{display:flex;justify-content:space-between;align-items:center;margin-top:auto}.current-price{font-size:1.8em;font-weight:700;color:var(--accent-color)}.store-link a{font-weight:600;color:var(--primary-color);text-decoration:none}#suggestions{margin-top:20px;text-align:center}#suggestions h3{margin-bottom:10px}#suggestions button{background-color:#e0e0e0;border:none;padding:8px 15px;margin:5px;border-radius:8px;cursor:pointer}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas por texto?</label><input type="text" id="query" name="query" placeholder="Ej: iPhone 15 Pro, red"></div><div class="input-group"><label for="image_file">... o mejora tu búsqueda subiendo una imagen</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Buscando las mejores ofertas...</p></div><section id="results-section" class="results-section"><h2 id="results-title">Mejores Ofertas Encontradas</h2><div id="suggestions"></div><div id="products-grid" class="products-grid"></div></section></div>
<script>
const searchForm = document.getElementById("search-form"), queryInput = document.getElementById("query"), imageInput = document.getElementById("image_file"), loadingDiv = document.getElementById("loading"), resultsSection = document.getElementById("results-section"), productsGrid = document.getElementById("products-grid"), suggestionsDiv = document.getElementById("suggestions");
function performSearch() {
    const formData = new FormData(searchForm);
    loadingDiv.style.display = "block", resultsSection.style.display = "none", productsGrid.innerHTML = "", suggestionsDiv.innerHTML = "";
    fetch("{{ url_for('api_search') }}", { method: "POST", body: formData }).then(response => response.json()).then(data => {
        loadingDiv.style.display = "none";
        if (data.results && data.results.length > 0) {
            document.getElementById('results-title').textContent = "Mejores Ofertas Encontradas";
            data.results.forEach(product => {
                productsGrid.innerHTML += `
                    <div class="product-card">
                        <div class="product-image"><img src="${product.image_url || 'https://via.placeholder.com/300'}" alt="${product.name}" onerror="this.onerror=null;this.src='https://via.placeholder.com/300';"></div>
                        <div class="product-info">
                            <div class="product-title">${product.name}</div>
                            <div class="price-store-wrapper">
                                <div class="current-price">$${product.price.toFixed(2)}</div>
                                <div class="store-link"><a href="${product.url}" target="_blank">Ver en ${product.store}</a></div>
                            </div>
                        </div>
                    </div>`;
            });
        } else if (data.suggestions && data.suggestions.length > 0) {
            document.getElementById('results-title').textContent = "Resultados no encontrados";
            let suggestionsHTML = '<h3>No encontramos resultados. ¿Quizás quisiste decir...?</h3>';
            data.suggestions.forEach(suggestion => { suggestionsHTML += `<button class="suggestion-btn">${suggestion}</button>`; });
            suggestionsDiv.innerHTML = suggestionsHTML;
            document.querySelectorAll('.suggestion-btn').forEach(button => {
                button.addEventListener('click', () => {
                    queryInput.value = button.textContent, imageInput.value = "", document.getElementById("image-preview-container").style.display = "none", performSearch();
                });
            });
        } else {
            document.getElementById('results-title').textContent = "Resultados no encontrados";
            productsGrid.innerHTML = "<p>No se encontraron resultados para tu búsqueda.</p>";
        }
        resultsSection.style.display = "block";
    }).catch(error => {
        console.error("Error:", error), loadingDiv.style.display = "none", productsGrid.innerHTML = "<p>Ocurrió un error durante la búsqueda. Por favor, intenta de nuevo.</p>", resultsSection.style.display = "block";
    });
}
searchForm.addEventListener("submit", function(e) { e.preventDefault(), performSearch(); });
imageInput.addEventListener("change", function() { if (this.files && this.files[0]) { var reader = new FileReader(); reader.onload = function(e) { document.getElementById("image-preview").src = e.target.result, document.getElementById("image-preview-container").style.display = "flex"; }, reader.readAsDataURL(this.files[0]); } });
document.getElementById("remove-image-btn").addEventListener("click", function() { imageInput.value = "", document.getElementById("image-preview").src = "#", document.getElementById("image-preview-container").style.display = "none"; });
</script>
</body></html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
