# app.py (versión 14.6 - Modo de Calidad)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 14.6 (Quality Focus Mode)
# Novedades:
# - NUEVO: Se reintroduce la IA para mejorar la calidad de los resultados de forma eficiente.
# - La consulta del usuario ahora se "mejora" con Gemini para hacerla más específica para compras.
# - La verificación por IA de si una página es de un producto se reactiva, pero solo se usa si se encuentra un precio, para ahorrar cuota de API.
# - Se implementa una categorización simple por palabras clave (sin IA) para decidir cuándo filtrar marketplaces.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
import io
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash
from PIL import Image

# --- IMPORTS DE APIS DE GOOGLE ---
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    print("✅ Módulo de Google Generative AI (Gemini) importado.")
except ImportError:
    print("⚠️ AVISO: 'google-generativeai' no está instalado.")
    genai = None
    google_exceptions = None

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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (HÍBRIDA Y EXPERTA)
# ==============================================================================

def _deep_scrape_content(url: str) -> Dict[str, Any]:
    headers = {'User-Agent': UserAgent().random, 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/'}
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        price_text = "N/A"
        price_selectors = ['[class*="price"]', '[id*="price"]', '[class*="Price"]', '[id*="Price"]', '[itemprop="price"]']
        for selector in price_selectors:
            price_tag = soup.select_one(selector)
            if price_tag:
                match = re.search(r'\d{1,3}(?:,?\d{3})*(?:\.\d{2})?', price_tag.get_text())
                if match: price_text = match.group(0).replace(',', ''); break
        image_url = ""
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"): image_url = urljoin(url, og_image["content"])
        title = soup.title.string.strip() if soup.title else 'Sin título'
        text_content = ' '.join(soup.stripped_strings)[:1500]
        return {'title': title, 'text': text_content, 'price': price_text, 'image': image_url}
    except Exception:
        return {'title': 'N/A', 'text': '', 'price': 'N/A', 'image': ''}

# --- NUEVA FUNCIÓN DE IA PARA MEJORAR LA CONSULTA ---
def _enhance_query_for_purchase(text: str, errors_list: List[str]) -> str:
    """Usa Gemini para convertir una consulta simple en una consulta de compra detallada en inglés."""
    if not genai or not text: return text
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"A user wants to buy a product. Enhance and translate their search query into a specific, detailed English query suitable for finding the product for sale online. Include relevant keywords like size, type, or 'for sale'. User query: '{text}'. Respond ONLY with the enhanced English query."
        response = model.generate_content(prompt)
        enhanced_query = response.text.strip()
        print(f"  🧠 Consulta mejorada por IA: de '{text}' a '{enhanced_query}'.")
        return enhanced_query
    except google_exceptions.ResourceExhausted as e:
        error_msg = "Advertencia: Se superó la cuota de API de IA para mejorar la consulta. Usando texto original."
        print(f"  ❌ {error_msg}")
        if error_msg not in errors_list: errors_list.append(error_msg)
        return text
    except Exception as e:
        print(f"  ❌ Error al mejorar la consulta: {e}. Usando texto original.")
        return text

# --- FUNCIÓN DE VERIFICACIÓN DE IA REACTIVADA ---
def _verify_is_product_page(query: str, page_title: str, page_content: str, errors_list: List[str]) -> bool:
    """Verifica si una página es una página de producto real usando IA."""
    if not genai: return True
    prompt_template = (f"You are a strict verification analyst. Based on the user's search and the page content, is this a direct retail page for the main product, and not just an accessory, a blog post, a review, or a forum discussion? User search: '{query}'. Page title: '{page_title}'. Page content snippet: '{page_content[:500]}'. Answer ONLY with YES or NO.")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt_template)
        is_product_page = "YES" in response.text.strip().upper()
        if not is_product_page: print(f"  🤖 IA descartó página '{page_title}' por no ser de producto.")
        return is_product_page
    except google_exceptions.ResourceExhausted as e:
        error_msg = "Advertencia: Se superó la cuota de API de IA para la verificación de páginas. Los resultados pueden ser menos precisos."
        print(f"  ❌ {error_msg}")
        if error_msg not in errors_list: errors_list.append(error_msg)
        return False # Es más seguro descartar si no podemos verificar
    except Exception: 
        return False

# --- CATEGORIZACIÓN SIMPLE POR PALABRAS CLAVE (SIN IA) ---
def _get_simple_category(query: str) -> str:
    """Determina si la consulta es para 'hardware/industrial' basado en palabras clave."""
    hardware_keywords = ['tape', 'part', 'tool', 'engine', 'motor', 'bearing', 'screw', 'bolt', 'industrial', 'liquidators', 'supply', 'hardware']
    if any(keyword in query.lower() for keyword in hardware_keywords):
        print("  🔩 Categoría detectada: hardware/industrial (por palabra clave).")
        return 'hardware_industrial'
    return 'consumer_tech'

def _get_clean_company_name(item: Dict) -> str:
    try:
        if source := item.get('source'): return source
        return urlparse(item.get('link', '')).netloc.replace('www.', '').split('.')[0].capitalize()
    except: return "Tienda"

@dataclass
class ProductResult:
    name: str; price: float; store: str; url: str; image_url: str = ""

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key

    def get_descriptive_query_from_image(self, image_content: bytes, errors_list: List[str]) -> Optional[str]:
        # (Sin cambios, esta función ya es de alta calidad)
        if not genai: print("  ❌ Análisis con Gemini Vision saltado."); return None
        print("  🧠 Analizando imagen con Gemini Vision...")
        try:
            image_pil = Image.open(io.BytesIO(image_content))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = """You are an expert product identifier. Analyze the image and generate a single, effective English search query to find this product online. Respond ONLY with the search query."""
            response = model.generate_content([prompt, image_pil])
            query = response.text.strip().replace("*", "")
            print(f"  ✅ Consulta generada por Gemini Vision: '{query}'")
            return query
        except google_exceptions.ResourceExhausted as e:
            error_msg = "Advertencia: Se ha superado la cuota de la API de IA. No se pudo analizar la imagen."
            print(f"  ❌ {error_msg}")
            if error_msg not in errors_list: errors_list.append(error_msg)
            return None
        except Exception as e:
            print(f"  ❌ Fallo en análisis con Gemini Vision: {e}"); return None

    def search_google_shopping(self, query: str) -> List[ProductResult]:
        print(f"--- Iniciando búsqueda en Google Shopping para: '{query}' ---")
        params = {"q": query, "engine": "google_shopping", "location": "United States", "gl": "us", "hl": "en", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            response.raise_for_status()
            products = []
            shopping_results = response.json().get('shopping_results', [])
            if not isinstance(shopping_results, list): return []
            for item in shopping_results:
                if isinstance(item, dict) and item.get('title') and item.get('price') and item.get('link'):
                    try:
                        price_str = item.get('extracted_price') or item.get('price')
                        price_float = float(re.sub(r'[^\d.]', '', str(price_str)))
                        if price_float >= 0.50:
                            products.append(ProductResult(
                                name=item['title'], price=price_float, store=item.get('source', 'Google'),
                                url=item['link'], image_url=item.get('thumbnail', '')
                            ))
                    except (ValueError, TypeError, KeyError): continue
            print(f"✅ Google Shopping encontró {len(products)} resultados válidos.")
            return products
        except Exception as e:
            print(f"❌ Ocurrió un error en Google Shopping: {e}"); return []

    def search_with_ai_verification(self, query: str, category: str, errors_list: List[str]) -> List[ProductResult]:
        print(f"--- Iniciando búsqueda profunda de calidad ({category}): '{query}' ---")
        params = {"q": query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "20", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            initial_results = response.json().get('organic_results', [])
            
            # Lista negra más agresiva para resultados de alta calidad
            blacklist = ['amazon.com', 'walmart.com', 'ebay.com', 'pinterest.com', 'youtube.com', 'wikipedia.org']
            if category == 'hardware_industrial':
                print("  ℹ️  Aplicando filtro de marketplaces para búsqueda de hardware.")
            else:
                blacklist = ['pinterest.com', 'youtube.com', 'wikipedia.org'] # Menos restrictivo para productos de consumo
                
            filtered_results = [item for item in initial_results if isinstance(item, dict) and not any(site in item.get('link', '') for site in blacklist)]
            valid_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_item = {executor.submit(_deep_scrape_content, item.get('link')): item for item in filtered_results if item.get('link')}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    content = future.result()
                    if content and content['price'] != "N/A":
                        # ¡LÓGICA CLAVE! Solo verificamos con IA si encontramos un precio.
                        if _verify_is_product_page(query, content['title'], content['text'], errors_list):
                            try:
                                price_float = float(content['price'])
                                if price_float >= 0.50:
                                    valid_results.append(ProductResult(name=content['title'], price=price_float, store=_get_clean_company_name(item), url=item.get('link'), image_url=content['image'] or item.get('thumbnail', '')))
                            except (ValueError, TypeError): continue
            print(f"✅ Búsqueda profunda encontró {len(valid_results)} resultados verificados.")
            return valid_results
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda profunda: {e}"); return []

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str], List[str]]:
        errors_list = []
        final_query = None
        original_query = query.strip() if query else None
        
        if original_query:
            final_query = _enhance_query_for_purchase(original_query, errors_list)
        
        if image_content:
            image_query = self.get_descriptive_query_from_image(image_content, errors_list)
            if image_query:
                # Si hay texto e imagen, combina la consulta mejorada con la de la imagen.
                final_query = f"{final_query} {image_query}" if final_query else image_query
        
        if not final_query: 
            print("❌ No se pudo determinar una consulta válida.")
            return [], [], errors_list
        
        category = _get_simple_category(original_query or final_query)
        print(f"🔍 Lanzando búsqueda HÍBRIDA (Modo Calidad, Cat: {category}) para: '{final_query}'")
        
        all_results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_deep_search = executor.submit(self.search_with_ai_verification, final_query, category, errors_list)
            future_shopping_search = executor.submit(self.search_google_shopping, final_query)
            all_results.extend(future_deep_search.result())
            all_results.extend(future_shopping_search.result())

        if not all_results:
            print("🤔 No se encontraron resultados.")
            return [], [], errors_list

        seen_urls = set()
        unique_results = []
        for product in all_results:
            if product.url not in seen_urls:
                unique_results.append(product)
                seen_urls.add(product.url)
        
        unique_results.sort(key=lambda x: x.price)
        print(f"✅ Búsqueda finalizada. {len(unique_results)} resultados únicos encontrados.")
        return unique_results, [], errors_list

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
    
    results, suggestions, errors = shopping_bot.search_product(query=query, image_content=image_content)
    
    results_dicts = [res.__dict__ for res in results]
    return jsonify(results=results_dicts, suggestions=suggestions, errors=errors)

# ==============================================================================
# SECCIÓN 4: PLANTILLAS HTML Y EJECUCIÓN
# ==============================================================================
AUTH_TEMPLATE_LOGIN_ONLY = """
<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Acceso | Smart Shopping Bot</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--text-color-dark:#2C3E50;--card-bg:#FFFFFF;--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:linear-gradient(135deg,var(--primary-color) 0%,var(--secondary-color) 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}.auth-container{max-width:480px;width:100%;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-medium);overflow:hidden;animation:fadeIn .8s ease-out}@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.form-header{text-align:center;padding:40px 30px 20px}.form-header h1{color:var(--text-color-dark);font-size:2em;margin-bottom:10px}.form-header p{color:#7f8c8d;font-size:1.1em}.form-body{padding:10px 40px 40px}form{display:flex;flex-direction:column;gap:20px}.input-group{display:flex;flex-direction:column;gap:8px}.input-group label{font-weight:600;color:var(--text-color-dark);font-size:.95em}.input-group input{padding:16px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:16px;transition:all .3s ease}.input-group input:focus{outline:0;border-color:var(--primary-color);box-shadow:0 0 0 4px rgba(74,144,226,.2)}.submit-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:16px 30px;font-size:1.1em;font-weight:600;border-radius:12px;cursor:pointer;transition:all .3s ease;margin-top:15px}.submit-btn:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(0,0,0,.2)}.flash-messages{list-style:none;padding:0 40px 20px}.flash{padding:15px;margin-bottom:15px;border-radius:8px;text-align:center}.flash.success{background-color:#d4edda;color:#155724}.flash.danger{background-color:#f8d7da;color:#721c24}.flash.warning{background-color:#fff3cd;color:#856404}</style></head><body><div class="auth-container"><div class="form-header"><h1>Bienvenido de Nuevo</h1><p>Accede para encontrar las mejores ofertas.</p></div>{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}<ul class=flash-messages>{% for category, message in messages %}<li class="flash {{ category }}">{{ message }}</li>{% endfor %}</ul>{% endif %}{% endwith %}<div class="form-body"><form id="login-form" action="{{ url_for('login') }}" method="post"><div class="input-group"><label for="login-email">Correo</label><input type="email" name="email" required></div><div class="input-group"><label for="login-password">Contraseña</label><input type="password" name="password" required></div><button type="submit" class="submit-btn">Entrar</button></form></div></div></body></html>
"""
SEARCH_TEMPLATE = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.api-errors{background-color:#fff3cd;color:#856404;padding:20px;border-radius:12px;margin-bottom:30px;text-align:left;border:1px solid #ffeeba}.api-errors ul{padding-left:20px;margin:0}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee;display:flex;flex-direction:column;position:relative}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center;overflow:hidden}.product-image img{width:100%;height:100%;object-fit:cover}.product-info{padding:25px;display:flex;flex-direction:column;flex-grow:1;justify-content:space-between}.product-title{font-size:1.1em;font-weight:600;margin-bottom:12px;color:var(--text-color-dark)}.price-store-wrapper{display:flex;justify-content:space-between;align-items:center;margin-top:auto}.current-price{font-size:1.8em;font-weight:700;color:var(--accent-color)}.store-link a{font-weight:600;color:var(--primary-color);text-decoration:none}#suggestions{margin-top:20px;text-align:center}#suggestions h3{margin-bottom:10px}#suggestions button{background-color:#e0e0e0;border:none;padding:8px 15px;margin:5px;border-radius:8px;cursor:pointer}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas por texto?</label><input type="text" id="query" name="query" placeholder="Ej: cinta de pintor azul 2 pulgadas"></div><div class="input-group"><label for="image_file">... o mejora tu búsqueda subiendo una imagen</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Realizando búsqueda de alta calidad...</p></div><section id="results-section" class="results-section"><div id="api-errors" class="api-errors" style="display:none;"></div><h2 id="results-title">Mejores Ofertas Encontradas</h2><div id="suggestions"></div><div id="products-grid" class="products-grid"></div></section></div>
<script>
const searchForm = document.getElementById("search-form"), queryInput = document.getElementById("query"), imageInput = document.getElementById("image_file"), loadingDiv = document.getElementById("loading"), resultsSection = document.getElementById("results-section"), productsGrid = document.getElementById("products-grid"), suggestionsDiv = document.getElementById("suggestions"), apiErrorsDiv = document.getElementById("api-errors");
function performSearch() {
    const formData = new FormData(searchForm);
    loadingDiv.style.display = "block";
    resultsSection.style.display = "none";
    productsGrid.innerHTML = "";
    suggestionsDiv.innerHTML = "";
    apiErrorsDiv.innerHTML = "";
    apiErrorsDiv.style.display = "none";

    fetch("{{ url_for('api_search') }}", { method: "POST", body: formData }).then(response => response.json()).then(data => {
        loadingDiv.style.display = "none";

        if (data.errors && data.errors.length > 0) {
            let errorHTML = '<strong>Advertencias de la Búsqueda:</strong><ul>';
            data.errors.forEach(error => { errorHTML += `<li>${error}</li>`; });
            errorHTML += '</ul><p>Nota: Los errores de cuota pueden reducir la calidad de los resultados. Considera actualizar tu plan de API de Google.</p>';
            apiErrorsDiv.innerHTML = errorHTML;
            apiErrorsDiv.style.display = "block";
        }

        if (data.results && data.results.length > 0) {
            document.getElementById("results-title").style.display = "block";
            productsGrid.innerHTML = ""; // Limpiar antes de añadir
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
        } else {
            document.getElementById("results-title").style.display = "none";
            if (!apiErrorsDiv.innerHTML) { 
                 productsGrid.innerHTML = "<p>No se encontraron resultados para tu búsqueda.</p>";
            }
        }
        resultsSection.style.display = "block";
    }).catch(error => {
        console.error("Error:", error);
        loadingDiv.style.display = "none";
        productsGrid.innerHTML = "<p>Ocurrió un error crítico durante la búsqueda. Por favor, revisa los logs del servidor.</p>";
        resultsSection.style.display = "block";
    });
}
searchForm.addEventListener("submit", function(e) { e.preventDefault(), performSearch(); });
imageInput.addEventListener("change", function() { if (this.files && this.files[0]) { var reader = new FileReader(); reader.onload = function(e) { document.getElementById("image-preview").src = e.target.result; document.getElementById("image-preview-container").style.display = "flex"; }; reader.readAsDataURL(this.files[0]); } });
document.getElementById("remove-image-btn").addEventListener("click", function() { imageInput.value = ""; document.getElementById("image-preview").src = "#"; document.getElementById("image-preview-container").style.display = "none"; });
</script>
</body></html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
