# app.py (versión final con HTML restaurado)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 5.1 (AI-Powered Search, HTML Fixed)
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
import statistics
from bs4 import BeautifulSoup

# --- IMPORTS DE FLASK Y SEGURIDAD ---
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash

# --- IMPORTACIÓN DE GOOGLE CLOUD VISION ---
try:
    from google.cloud import vision
    print("✅ Módulo de Google Cloud Vision importado.")
except ImportError:
    print("❌ ERROR: El módulo 'google-cloud-vision' no está instalado.")
    vision = None

# --- IMPORTACIÓN DE GOOGLE GENERATIVE AI (GEMINI) ---
try:
    import google.generativeai as genai
    print("✅ Módulo de Google Generative AI (Gemini) importado.")
except ImportError:
    print("⚠️ AVISO: El módulo 'google-generativeai' no está instalado.")
    genai = None

# =============== CONFIGURACIÓN DE APIs MEDIANTE VARIABLES DE ENTORNO ===============

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get('GOOGLE_CREDENTIALS_JSON')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR: No se pudo configurar la API de Gemini: {e}")
        genai = None

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
# SECCIÓN 1: LÓGICA DEL SMART SHOPPING BOT
# ==============================================================================
def _deep_scrape_content(url: str) -> Dict:
    headers = {'User-Agent': UserAgent().random}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        price_text = "N/A"
        price_tags = soup.find_all(text=re.compile(r'\$\s?\d+([.,]\d+)?'))
        if price_tags:
            match = re.search(r'(\d+([.,]\d+)?)', price_tags[0])
            if match: price_text = match.group(0).replace(',', '')
        title = soup.title.string if soup.title else 'Sin título'
        text_content = ' '.join(soup.stripped_strings)[:1000]
        print(f"    Scraping en {url[:40]}... Título: '{title[:30]}...', Precio: {price_text}")
        return {'title': title, 'text': text_content, 'price': price_text}
    except Exception as e:
        print(f"    Falló el scraping en {url[:40]}: {e}")
        return {'title': 'N/A', 'text': '', 'price': 'N/A'}

def _verify_product_with_gemini(query: str, product_title: str, product_text: str) -> bool:
    if not genai:
        print("    (Saltando verificación IA: Gemini no configurado)")
        return True
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Is the following product relevant to the user's search? Answer only with 'SI' or 'NO'. User search: "{query}". Page title: "{product_title}". Page text extract: "{product_text[:500]}" """
        response = model.generate_content(prompt)
        decision = response.text.strip().upper()
        print(f"    Verificación IA para '{product_title[:30]}...': {decision}")
        return "SI" in decision
    except Exception as e:
        print(f"    Error en verificación IA: {e}")
        return False

def _get_clean_company_name(item: Dict) -> str:
    try:
        if source := item.get('source'): return source
        return urlparse(item.get('link', '')).netloc.replace('www.', '').split('.')[0].capitalize()
    except:
        return "Tienda"

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
        else: self.vision_client = None
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
        if image_content: final_query = self.get_query_from_image_vision_api(image_content)
        if not final_query: print("❌ No se pudo determinar una consulta válida."); return [], False
        print(f"🔍 Lanzando búsqueda AVANZADA para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query)
        return best_deals, False
    def search_with_ai_verification(self, search_query: str) -> List[ProductResult]:
        print(f"--- Iniciando búsqueda de PRODUCTOS para: '{search_query}' ---")
        params = {"q": search_query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "50", "api_key": self.serpapi_key}
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
                            results_with_price.append({'store': store_name, 'product_name': item.get('title', 'Sin título'), 'price_float': float(content['price'].replace(',', '')), 'url': item.get('link'), 'image_url': item.get('thumbnail', '')})
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
# SECCIÓN 2: CONFIGURACIÓN DE FLASK Y RUTAS
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

## GÉNESIS: ¡CÓDIGO HTML COMPLETO RESTAURADO! ##
AUTH_TEMPLATE_LOGIN_ONLY = """
<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Acceso | Smart Shopping Bot</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--text-color-dark:#2C3E50;--card-bg:#FFFFFF;--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:linear-gradient(135deg,var(--primary-color) 0%,var(--secondary-color) 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}.auth-container{max-width:480px;width:100%;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-medium);overflow:hidden;animation:fadeIn .8s ease-out}@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.form-header{text-align:center;padding:40px 30px 20px}.form-header h1{color:var(--text-color-dark);font-size:2em;margin-bottom:10px}.form-header p{color:#7f8c8d;font-size:1.1em}.form-body{padding:10px 40px 40px}form{display:flex;flex-direction:column;gap:20px}.input-group{display:flex;flex-direction:column;gap:8px}.input-group label{font-weight:600;color:var(--text-color-dark);font-size:.95em}.input-group input{padding:16px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:16px;transition:all .3s ease}.input-group input:focus{outline:0;border-color:var(--primary-color);box-shadow:0 0 0 4px rgba(74,144,226,.2)}.submit-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:16px 30px;font-size:1.1em;font-weight:600;border-radius:12px;cursor:pointer;transition:all .3s ease;margin-top:15px}.submit-btn:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(0,0,0,.2)}.flash-messages{list-style:none;padding:0 40px 20px}.flash{padding:15px;margin-bottom:15px;border-radius:8px;text-align:center}.flash.success{background-color:#d4edda;color:#155724}.flash.danger{background-color:#f8d7da;color:#721c24}.flash.warning{background-color:#fff3cd;color:#856404}</style></head><body><div class="auth-container"><div class="form-header"><h1>Bienvenido de Nuevo</h1><p>Accede para encontrar las mejores ofertas.</p></div>{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}<ul class=flash-messages>{% for category, message in messages %}<li class="flash {{ category }}">{{ message }}</li>{% endfor %}</ul>{% endif %}{% endwith %}<div class="form-body"><form id="login-form" action="{{ url_for('login') }}" method="post"><div class="input-group"><label for="login-email">Correo</label><input type="email" name="email" required></div><div class="input-group"><label for="login-password">Contraseña</label><input type="password" name="password" required></div><button type="submit" class="submit-btn">Entrar</button></form></div></div></body></html>
"""

SEARCH_TEMPLATE = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center}.product-image img{max-width:90%;max-height:90%;object-fit:contain}.product-info{padding:25px}.product-title{font-size:1.25em;font-weight:600;margin-bottom:12px}.current-price{font-size:2.2em;font-weight:700;color:var(--accent-color)}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas por texto?</label><input type="text" id="query" name="query" placeholder="Ej: iPhone 15 Pro"></div><div class="input-group"><label for="image_file">... o busca subiendo una imagen</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Buscando las mejores ofertas...</p></div><section id="results-section" class="results-section"><h2 id="results-title">Mejores Ofertas Encontradas</h2><div id="products-grid" class="products-grid"></div></section></div><script>const searchForm=document.getElementById("search-form");searchForm.addEventListener("submit",function(e){e.preventDefault();const t=new FormData(searchForm);document.getElementById("loading").style.display="block",document.getElementById("results-section").style.display="none",fetch("{{ url_for('api_search') }}",{method:"POST",body:t}).then(e=>e.json()).then(e=>{document.getElementById("loading").style.display="none";const t=document.getElementById("products-grid");if(t.innerHTML="",e.results&&e.results.length>0)e.results.forEach(e=>{t.innerHTML+=`\n                            <div class="product-card">\n                                <div class="product-image"><img src="${e.image_url||""}" alt="${e.name}"></div>\n                                <div class="product-info">\n                                    <div class="product-title">${e.name}</div>\n                                    <div class="current-price">$${e.price.toFixed(2)}</div>\n                                    <div><a href="${e.url}" target="_blank">Ver en ${e.store}</a></div>\n                                </div>\n                            </div>`});else t.innerHTML="<p>No se encontraron resultados.</p>";document.getElementById("results-section").style.display="block"}).catch(e=>{console.error("Error:",e),document.getElementById("loading").style.display="none"})}),document.getElementById("image_file").addEventListener("change",function(){if(this.files&&this.files[0]){var e=new FileReader;e.onload=function(e){document.getElementById("image-preview").src=e.target.result,document.getElementById("image-preview-container").style.display="flex"},e.readAsDataURL(this.files[0])}}),document.getElementById("remove-image-btn").addEventListener("click",function(){document.getElementById("image_file").value="",document.getElementById("image-preview").src="#",document.getElementById("image-preview-container").style.display="none"});</script></body></html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
