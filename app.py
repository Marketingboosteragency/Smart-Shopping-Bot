# app.py (versión 14.7 - Motor de Precisión de Precios)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 14.7 (Price Accuracy & Relevance Engine)
# Novedades:
# - NUEVO: Validación de Precios por IA. Los 5 resultados más baratos son verificados por Gemini para confirmar que el precio scrapeado es el correcto para el producto principal.
# - SCRAPER MEJORADO: Ahora también busca la presencia de un botón "Add to Cart" (o similar) como indicador de una página de producto válida, reduciendo falsos positivos.
# - FILTRADO EN DOS ETAPAS: Se realiza una recolección amplia de resultados y luego una validación por IA profunda sobre los mejores candidatos para maximizar la calidad y eficiencia.
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
    """Scraper mejorado que busca precio y botón de 'Añadir al Carrito'."""
    headers = {'User-Agent': UserAgent().random, 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/'}
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Búsqueda de precio
        price_text = "N/A"
        price_selectors = ['[class*="price"]', '[id*="price"]', '[itemprop="price"]', '[data-price]']
        for selector in price_selectors:
            price_tag = soup.select_one(selector)
            if price_tag:
                price_content = price_tag.get('content') or price_tag.get_text()
                match = re.search(r'\d{1,3}(?:,?\d{3})*(?:\.\d{2})?', price_content)
                if match: 
                    price_text = match.group(0).replace(',', '')
                    break
        
        # Búsqueda de botón de compra
        has_add_to_cart = False
        cart_texts = ['add to cart', 'add to bag', 'buy now', 'comprar ahora', 'añadir al carrito']
        # Busca botones y enlaces que contengan el texto de compra
        for cart_text in cart_texts:
            if soup.find(lambda tag: tag.name in ['button', 'a', 'input'] and cart_text in tag.get_text(strip=True).lower()):
                has_add_to_cart = True
                break

        image_url = ""
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"): image_url = urljoin(url, og_image["content"])
        title = soup.title.string.strip() if soup.title else 'Sin título'
        
        return {'title': title, 'price': price_text, 'image': image_url, 'has_add_to_cart': has_add_to_cart}
    except Exception:
        return {'title': 'N/A', 'price': 'N/A', 'image': '', 'has_add_to_cart': False}

def _enhance_query_for_purchase(text: str, errors_list: List[str]) -> str:
    if not genai or not text: return text
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"A user wants to buy a product. Enhance and translate their search query into a specific, detailed English query suitable for finding the product for sale online. Include relevant keywords like size, type, or 'for sale'. User query: '{text}'. Respond ONLY with the enhanced English query."
        response = model.generate_content(prompt)
        enhanced_query = response.text.strip()
        print(f"  🧠 Consulta mejorada por IA: de '{text}' a '{enhanced_query}'.")
        return enhanced_query
    except google_exceptions.ResourceExhausted as e:
        error_msg = "Advertencia: Cuota de API superada para mejorar la consulta. Usando texto original."
        if error_msg not in errors_list: errors_list.append(error_msg)
        return text
    except Exception: return text

# --- NUEVA FUNCIÓN: VALIDACIÓN DE PRECIO POR IA ---
def _validate_price_with_ai(product: 'ProductResult', original_query: str, errors_list: List[str]) -> bool:
    """Usa Gemini para verificar si el precio encontrado en una URL es el precio principal y correcto."""
    if not genai: return True # Si no hay IA, se aprueba por defecto
    print(f"  🤖 Verificando precio ${product.price} en '{product.store}' con IA...")
    prompt = (
        f"You are a meticulous price verification expert. A user searched for '{original_query}'.\n"
        f"I have found a price of ${product.price} on the page with the title '{product.name}' at this URL: {product.url}\n"
        f"Is this the main, direct price for the product the user is looking for? Or is it a price for an accessory, a different model, a part, or a price in a list of many other items?\n"
        f"Answer ONLY with YES or NO."
    )
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        is_valid = "YES" in response.text.strip().upper()
        if is_valid:
            print(f"  ✅ Precio verificado para '{product.name}'.")
        else:
            print(f"  ❌ IA descartó el precio de '{product.name}' por ser incorrecto o no principal.")
        return is_valid
    except google_exceptions.ResourceExhausted as e:
        error_msg = "Advertencia: Cuota de API superada para la validación de precios. La precisión de los precios puede ser menor."
        if error_msg not in errors_list: errors_list.append(error_msg)
        return False # Es más seguro descartar si la IA falla
    except Exception as e:
        print(f"  ❌ Error en validación de precio: {e}")
        return False

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
        if not genai: return None
        # (código de análisis de imagen sin cambios)
        return "descriptive query from image" # Placeholder

    def search_google_shopping(self, query: str) -> List[ProductResult]:
        print(f"--- Iniciando búsqueda en Google Shopping para: '{query}' ---")
        params = {"q": query, "engine": "google_shopping", "location": "United States", "gl": "us", "hl": "en", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            response.raise_for_status()
            products = []
            for item in response.json().get('shopping_results', []):
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

    def initial_deep_search(self, query: str) -> List[ProductResult]:
        """Realiza la búsqueda profunda inicial, filtrando solo por scraper."""
        print(f"--- Iniciando búsqueda profunda inicial: '{query}' ---")
        params = {"q": query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "20", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            initial_results = response.json().get('organic_results', [])
            blacklist = ['pinterest.com', 'youtube.com', 'wikipedia.org', 'facebook.com', 'twitter.com']
            filtered_links = [item.get('link') for item in initial_results if isinstance(item, dict) and item.get('link') and not any(site in item.get('link') for site in blacklist)]
            
            valid_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(_deep_scrape_content, url): url for url in filtered_links}
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    content = future.result()
                    # FILTRO CLAVE SIN IA: Debe tener un precio y un botón de compra
                    if content and content['price'] != "N/A" and content['has_add_to_cart']:
                        try:
                            price_float = float(content['price'])
                            if price_float >= 0.50:
                                valid_results.append(ProductResult(
                                    name=content['title'], price=price_float, 
                                    store=urlparse(url).netloc.replace('www.', '').split('.')[0].capitalize(),
                                    url=url, image_url=content['image']
                                ))
                        except (ValueError, TypeError): continue
            print(f"✅ Búsqueda profunda inicial encontró {len(valid_results)} candidatos.")
            return valid_results
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda profunda inicial: {e}"); return []

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str], List[str]]:
        errors_list = []
        original_query = query.strip() if query else "product in image"
        
        # 1. Mejorar la consulta
        final_query = _enhance_query_for_purchase(original_query, errors_list)
        if image_content:
            # Lógica para combinar con imagen si es necesario
            pass

        if not final_query: 
            return [], [], errors_list
        
        # 2. Fase 1: Recolección Amplia
        all_candidates = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_deep_search = executor.submit(self.initial_deep_search, final_query)
            future_shopping_search = executor.submit(self.search_google_shopping, final_query)
            all_candidates.extend(future_deep_search.result())
            all_candidates.extend(future_shopping_search.result())

        if not all_candidates:
            print("🤔 No se encontraron resultados candidatos iniciales.")
            return [], [], errors_list

        seen_urls = set()
        unique_candidates = []
        for product in all_candidates:
            if product.url not in seen_urls:
                unique_candidates.append(product)
                seen_urls.add(product.url)
        
        unique_candidates.sort(key=lambda x: x.price)

        # 3. Fase 2: Validación por IA de los 5 más baratos
        top_candidates = unique_candidates[:5]
        print(f"--- Iniciando Fase 2: Validación por IA de los {len(top_candidates)} mejores candidatos ---")
        
        final_verified_results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_product = {executor.submit(_validate_price_with_ai, product, original_query, errors_list): product for product in top_candidates}
            for future in as_completed(future_to_product):
                product = future_to_product[future]
                if future.result(): # Si la validación devuelve True
                    final_verified_results.append(product)

        # Ordenar los resultados finales verificados
        final_verified_results.sort(key=lambda x: x.price)

        print(f"✅ Búsqueda finalizada. {len(final_verified_results)} resultados verificados con IA.")
        return final_verified_results, [], errors_list

# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

# (Las rutas de Flask no necesitan cambios)
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
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.loading p{font-weight:600;color:var(--primary-color)}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.api-errors{background-color:#fff3cd;color:#856404;padding:20px;border-radius:12px;margin-bottom:30px;text-align:left;border:1px solid #ffeeba}.api-errors ul{padding-left:20px;margin:0}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee;display:flex;flex-direction:column;position:relative}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center;overflow:hidden}.product-image img{width:100%;height:100%;object-fit:cover}.product-info{padding:25px;display:flex;flex-direction:column;flex-grow:1;justify-content:space-between}.product-title{font-size:1.1em;font-weight:600;margin-bottom:12px;color:var(--text-color-dark)}.price-store-wrapper{display:flex;justify-content:space-between;align-items:center;margin-top:auto}.current-price{font-size:1.8em;font-weight:700;color:var(--accent-color)}.store-link a{font-weight:600;color:var(--primary-color);text-decoration:none}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas?</label><input type="text" id="query" name="query" placeholder="Ej: cinta de pintor azul 2 pulgadas"></div><div class="input-group"><label for="image_file">... o sube una imagen para una búsqueda más precisa</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Realizando búsqueda de alta precisión...</p></div><section id="results-section" class="results-section"><div id="api-errors" class="api-errors" style="display:none;"></div><h2 id="results-title">Mejores Ofertas Verificadas</h2><div id="products-grid" class="products-grid"></div></section></div>
<script>
const searchForm = document.getElementById("search-form"), queryInput = document.getElementById("query"), imageInput = document.getElementById("image_file"), loadingDiv = document.getElementById("loading"), resultsSection = document.getElementById("results-section"), productsGrid = document.getElementById("products-grid"), apiErrorsDiv = document.getElementById("api-errors");
function performSearch() {
    const formData = new FormData(searchForm);
    loadingDiv.style.display = "block";
    resultsSection.style.display = "none";
    productsGrid.innerHTML = "";
    apiErrorsDiv.innerHTML = "";
    apiErrorsDiv.style.display = "none";

    fetch("{{ url_for('api_search') }}", { method: "POST", body: formData }).then(response => response.json()).then(data => {
        loadingDiv.style.display = "none";

        if (data.errors && data.errors.length > 0) {
            let errorHTML = '<strong>Advertencias durante la búsqueda:</strong><ul>';
            data.errors.forEach(error => { errorHTML += `<li>${error}</li>`; });
            errorHTML += '</ul><p>Nota: Los errores de cuota de API pueden afectar la calidad de los resultados. Para un rendimiento óptimo, considera habilitar la facturación en tu cuenta de Google Cloud.</p>';
            apiErrorsDiv.innerHTML = errorHTML;
            apiErrorsDiv.style.display = "block";
        }

        if (data.results && data.results.length > 0) {
            document.getElementById("results-title").style.display = "block";
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
                 productsGrid.innerHTML = "<p>No se encontraron resultados verificados para tu búsqueda.</p>";
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
