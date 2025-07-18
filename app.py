# app.py (versión 23.0 - Motor de Búsqueda Secuencial y Priorizada)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 23.0 (Sequential & Prioritized Search Engine)
# Novedades:
# - CORRECCIÓN DEFINITIVA DE ERROR 429: Se reemplaza la búsqueda masiva en paralelo por una cascada secuencial y priorizada para respetar los límites de la API.
# - BÚSQUEDA EFICIENTE: El bot ahora busca primero en la fuente más probable (Google Shopping) y solo expande la búsqueda si es necesario, ahorrando llamadas a la API.
# - ARQUITECTURA ROBUSTA: Se mantiene el motor de producción endurecido y la lógica de fallback, ahora activada de forma más inteligente.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
import io
import traceback
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
    genai = None; google_exceptions = None

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL DE FLASK Y APIS
# ==============================================================================
app = Flask(__name__)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-muy-fuerte')

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR al configurar API de Gemini: {e}"); genai = None

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT
# ==============================================================================

@dataclass
class ProductResult:
    name: str; store: str; url: str; image_url: str = ""
    price_in_usd: float = 0.0; original_price: float = 0.0; original_currency: str = "USD"
    relevance_score: int = 0; price_accuracy_score: int = 0;
    reasoning: str = ""; is_alternative_suggestion: bool = False

CURRENCY_RATES_TO_USD = {"USD": 1.0, "DOP": 0.017, "MXN": 0.054, "CAD": 0.73, "EUR": 1.08, "GBP": 1.27}

def _deep_scrape_content(url: str) -> Dict[str, Any]:
    headers = {'User-Agent': UserAgent().random, 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/'}
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        image_url = (og.get("content") for og in [soup.find("meta", property="og:image")] if og)
        image_url = urljoin(url, next(image_url, ''))
        title = soup.title.string.strip() if soup.title else 'No Title'
        text_content = ' '.join(soup.stripped_strings)[:2000]
        return {'title': title, 'image': image_url, 'text_content': text_content, 'url': url}
    except Exception:
        return {'title': 'N/A', 'image': '', 'text_content': '', 'url': url}

def _enhance_query_for_purchase(text: str, errors_list: List[str]) -> str:
    if not genai or not text: return text
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Enhance and translate this user's query into a specific, detailed English query for finding a product online. Query: '{text}'. Respond ONLY with the enhanced query."
        response = model.generate_content(prompt)
        enhanced_query = response.text.strip()
        print(f"  🧠 Consulta mejorada por IA: de '{text}' a '{enhanced_query}'.")
        return enhanced_query
    except google_exceptions.ResourceExhausted as e:
        errors_list.append("Advertencia: Cuota de API superada para mejorar la consulta.")
        return text
    except Exception: return text

def _get_fallback_query_from_ai(original_query: str, errors_list: List[str]) -> Optional[str]:
    if not genai: return None
    print(f"  🤔 La búsqueda precisa de '{original_query}' falló. Generando una consulta flexible...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"A search for '{original_query}' yielded no results. Generate a single, slightly broader but still relevant English query that is likely to find similar products. Respond ONLY with the new query."
        response = model.generate_content(prompt)
        fallback_query = response.text.strip()
        print(f"  💡 Consulta flexible generada: '{fallback_query}'")
        return fallback_query
    except google_exceptions.ResourceExhausted as e:
        errors_list.append("Advertencia: Cuota de API superada al intentar generar sugerencias.")
        return None
    except Exception: return None

def _get_ai_analysis(candidate: Dict[str, Any], original_query: str, errors_list: List[str]) -> Dict[str, Any]:
    default_failure = {"relevance_score": 0, "price_accuracy_score": 0}
    if not genai or not candidate.get('text_content'): return default_failure
    
    print(f"  🤖⚖️ Calificando oferta: '{candidate['title']}'...")
    prompt = (
        f"You are a shopping expert AI. Analyze the product page data and return a JSON object with your ratings.\n\n"
        f"DATA:\n- User's Search: '{original_query}'\n- Page Title: '{candidate['title']}'\n- Page Text: '{candidate['text_content']}'\n\n"
        f"TASKS & SCORING:\n"
        f"1. **Extract Price & Currency:** Find the main product's price and its 3-letter currency code (e.g., 'USD', 'MXN'). Assume 'USD' if unclear.\n"
        f"2. **Relevance Score (1-10):** How closely does this product match the user's search? 10 is perfect. Below 5 is a different product.\n"
        f"3. **Price Accuracy Score (1-10):** How confident are you the price is correct for a single unit? 10 is very confident.\n"
        f"4. **US Centric Check:** Does this store operate in or ship to the USA? This is critical.\n\n"
        f"Return a JSON with these keys: `price` (float), `currency` (string), `relevance_score` (int), `price_accuracy_score` (int), `is_usa_centric` (boolean), `reasoning` (string)."
    )
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        analysis = json.loads(response.text)
        if not all(k in analysis for k in ["price", "currency", "relevance_score", "price_accuracy_score", "is_usa_centric"]): return default_failure
        print(f"  🧠 Calificación IA: Relevancia={analysis['relevance_score']}/10, Precisión={analysis['price_accuracy_score']}/10, Precio={analysis['price']} {analysis['currency']}, USA?={analysis['is_usa_centric']}")
        return analysis
    except (json.JSONDecodeError, google_exceptions.ResourceExhausted, ValueError) as e:
        if isinstance(e, google_exceptions.ResourceExhausted): errors_list.append("Advertencia: Cuota de API superada durante el análisis.")
        return default_failure
    except Exception: return default_failure

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.MAX_RESULTS_TO_RETURN = 30
        self.MINIMUM_RESULTS_TARGET = 10

    def _run_single_search_task(self, query: str, engine: str, start: int = 0) -> List[str]:
        urls = []
        params = {"q": query, "engine": engine, "location": "United States", "gl": "us", "hl": "en", "api_key": self.serpapi_key, "start": start}
        if engine == "google": params["num"] = "10"
        try:
            print(f"  📡 Ejecutando búsqueda: Engine={engine}, Query='{query[:50]}...'")
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            response.raise_for_status()
            results = response.json().get('organic_results', []) if engine == "google" else response.json().get('shopping_results', [])
            for item in results:
                if isinstance(item, dict) and item.get('link'): urls.append(item['link'])
        except Exception as e: print(f"❌ Error en sub-búsqueda ({engine}): {e}")
        return urls
    
    def _process_and_validate_candidates(self, candidate_urls: List[str], original_query: str, errors_list: List[str], is_fallback: bool = False) -> List[ProductResult]:
        blacklist = ['pinterest.com', 'youtube.com', 'wikipedia.org', 'facebook.com']
        filtered_urls = list(set([url for url in candidate_urls if not any(site in url for site in blacklist)]))
        
        print(f"--- {len(filtered_urls)} URLs candidatas pasarán a la fase de scrape y juicio. ---")
        if not filtered_urls: return []

        analyzed_products = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            scraped_candidates = list(executor.map(_deep_scrape_content, filtered_urls))
            candidates_for_judgement = [c for c in scraped_candidates if c['text_content'] and len(c['text_content']) > 50]
            print(f"--- FASE 2: Sometiendo a juicio de IA a {len(candidates_for_judgement)} candidatos. ---")
            
            future_to_candidate = {executor.submit(_get_ai_analysis, c, original_query, errors_list): c for c in candidates_for_judgement}
            for future in as_completed(future_to_candidate):
                candidate_data, analysis = future_to_candidate[future], future.result()
                if analysis.get('is_usa_centric', False) and analysis.get('relevance_score', 0) >= 5 and analysis.get('price_accuracy_score', 0) >= 5:
                    currency = analysis.get('currency', 'USD').upper(); rate = CURRENCY_RATES_TO_USD.get(currency)
                    if rate:
                        original_price = float(analysis.get('price', 99999)); price_in_usd = original_price * rate
                        if price_in_usd >= 0.50:
                            analyzed_products.append(ProductResult(
                                name=candidate_data['title'], store=urlparse(candidate_data['url']).netloc.replace('www.', '').split('.')[0].capitalize(),
                                url=candidate_data['url'], image_url=candidate_data['image'],
                                price_in_usd=price_in_usd, original_price=original_price, original_currency=currency,
                                relevance_score=analysis['relevance_score'], price_accuracy_score=analysis['price_accuracy_score'], 
                                reasoning=analysis.get('reasoning', ''), is_alternative_suggestion=is_fallback
                            ))
        return analyzed_products

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str], List[str]]:
        errors_list = []
        try:
            original_query = query.strip() if query else "product from image"
            if not original_query: return [], [], []

            enhanced_query = _enhance_query_for_purchase(original_query, errors_list)
            if not enhanced_query: return [], ["No se pudo generar una consulta válida."], errors_list
            
            # --- BÚSQUEDA SECUENCIAL Y PRIORIZADA ---
            all_urls = set()
            
            # Intento 1: Google Shopping (La mejor fuente)
            print("--- Iniciando Búsqueda Priorizada: Google Shopping ---")
            urls = self._run_single_search_task(enhanced_query, "google_shopping")
            for url in urls: all_urls.add(url)
            
            final_results = self._process_and_validate_candidates(list(all_urls), original_query, errors_list)

            # Si no tenemos suficientes resultados, expandimos la búsqueda
            if len(final_results) < self.MINIMUM_RESULTS_TARGET:
                print(f"--- Menos de {self.MINIMUM_RESULTS_TARGET} resultados. Expandiendo búsqueda... ---")
                
                # Búsqueda Orgánica
                for i in range(2): # 2 páginas
                    urls = self._run_single_search_task(enhanced_query, "google", start=i * 10)
                    for url in urls: all_urls.add(url)

                # Búsqueda Consolidada en Tiendas
                high_priority_stores = ["amazon.com", "walmart.com", "ebay.com", "target.com", "homedepot.com", "lowes.com"]
                store_query_part = " OR ".join([f"site:{store}" for store in high_priority_stores])
                store_query = f"({store_query_part}) \"{original_query}\""
                urls = self._run_single_search_task(store_query, "google")
                for url in urls: all_urls.add(url)
                
                final_results = self._process_and_validate_candidates(list(all_urls), original_query, errors_list)

            # --- INTENTO FINAL: BÚSQUEDA FLEXIBLE (FALLBACK) ---
            if not final_results:
                print("--- Búsqueda principal sin resultados. Iniciando Búsqueda Flexible. ---")
                fallback_query = _get_fallback_query_from_ai(original_query, errors_list)
                if fallback_query:
                    fallback_urls = self._run_single_search_task(fallback_query, "google_shopping")
                    final_results = self._process_and_validate_candidates(fallback_urls, original_query, errors_list, is_fallback=True)
                    if final_results:
                        errors_list.insert(0, "No encontramos resultados exactos. Pero aquí hay algunas opciones similares que podrían interesarte.")

            if not final_results: 
                print("✅ BÚSQUEDA COMPLETA. No se encontraron ofertas de alta calidad.")
                return [], [], errors_list
            
            final_results = sorted(final_results, key=lambda p: p.price_in_usd)
            
            print(f"✅ BÚSQUEDA COMPLETA. Se encontraron {len(final_results)} ofertas de calidad.")
            return final_results[:self.MAX_RESULTS_TO_RETURN], [], errors_list

        except Exception as e:
            print(f"‼️ ERROR CRÍTICO NO MANEJADO EN search_product: {e}")
            traceback.print_exc()
            errors_list.append("Ocurrió un error inesperado en el servidor.")
            return [], [], errors_list

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
    if not FIREBASE_WEB_API_KEY: flash('Servicio no configurado.', 'danger'); return redirect(url_for('index'))
    email, password = request.form.get('email'), request.form.get('password')
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {'email': email, 'password': password, 'returnSecureToken': True}
    try:
        response = requests.post(rest_api_url, json=payload); response.raise_for_status()
        user_data = response.json()
        session['user_id'] = user_data['localId']; session['user_name'] = user_data.get('displayName', email); session['id_token'] = user_data['idToken']
        flash('¡Has iniciado sesión correctamente!', 'success'); return redirect(url_for('main_app_page'))
    except requests.exceptions.HTTPError as e:
        error_message = e.response.json().get('error', {}).get('message', 'ERROR')
        flash('Correo o contraseña incorrectos.' if 'INVALID' in error_message else f'Error: {error_message}', 'danger'); return redirect(url_for('index'))
    except Exception as e: flash(f'Ocurrió un error inesperado: {e}', 'danger'); return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear(); flash('Has cerrado la sesión.', 'success'); return redirect(url_for('index'))

@app.route('/app')
def main_app_page():
    if 'user_id' not in session: flash('Debes iniciar sesión para acceder.', 'warning'); return redirect(url_for('index'))
    return render_template_string(SEARCH_TEMPLATE, user_name=session.get('user_name', 'Usuario'))

@app.route('/api/search', methods=['POST'])
def api_search():
    if 'user_id' not in session: return jsonify({'error': 'No autorizado'}), 401
    query = request.form.get('query')
    image_file = request.files.get('image_file')
    image_content = image_file.read() if image_file and image_file.filename != '' else None
    results, _, errors = shopping_bot.search_product(query=query, image_content=image_content)
    results_dicts = [res.__dict__ for res in results]
    return jsonify(results=results_dicts, suggestions=[], errors=errors)

# ==============================================================================
# SECCIÓN 4: PLANTILLAS HTML Y EJECUCIÓN
# ==============================================================================
AUTH_TEMPLATE_LOGIN_ONLY = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Acceso | Smart Shopping Bot</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--text-color-dark:#2C3E50;--card-bg:#FFFFFF;--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:linear-gradient(135deg,var(--primary-color) 0%,var(--secondary-color) 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}.auth-container{max-width:480px;width:100%;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-medium);overflow:hidden;animation:fadeIn .8s ease-out}@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.form-header{text-align:center;padding:40px 30px 20px}.form-header h1{color:var(--text-color-dark);font-size:2em;margin-bottom:10px}.form-header p{color:#7f8c8d;font-size:1.1em}.form-body{padding:10px 40px 40px}form{display:flex;flex-direction:column;gap:20px}.input-group{display:flex;flex-direction:column;gap:8px}.input-group label{font-weight:600;color:var(--text-color-dark);font-size:.95em}.input-group input{padding:16px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:16px;transition:all .3s ease}.input-group input:focus{outline:0;border-color:var(--primary-color);box-shadow:0 0 0 4px rgba(74,144,226,.2)}.submit-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:16px 30px;font-size:1.1em;font-weight:600;border-radius:12px;cursor:pointer;transition:all .3s ease;margin-top:15px}.submit-btn:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(0,0,0,.2)}.flash-messages{list-style:none;padding:0 40px 20px}.flash{padding:15px;margin-bottom:15px;border-radius:8px;text-align:center}.flash.success{background-color:#d4edda;color:#155724}.flash.danger{background-color:#f8d7da;color:#721c24}.flash.warning{background-color:#fff3cd;color:#856404}</style></head><body><div class="auth-container"><div class="form-header"><h1>Bienvenido de Nuevo</h1><p>Accede para encontrar las mejores ofertas.</p></div>{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}<ul class=flash-messages>{% for category, message in messages %}<li class="flash {{ category }}">{{ message }}</li>{% endfor %}</ul>{% endif %}{% endwith %}<div class="form-body"><form id="login-form" action="{{ url_for('login') }}" method="post"><div class="input-group"><label for="login-email">Correo</label><input type="email" name="email" required></div><div class="input-group"><label for="login-password">Contraseña</label><input type="password" name="password" required></div><button type="submit" class="submit-btn">Entrar</button></form></div></div></body></html>
"""

SEARCH_TEMPLATE = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.loading p{font-weight:600;color:var(--primary-color)}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.api-errors{background-color:#fff3cd;color:#856404;padding:20px;border-radius:12px;margin-bottom:30px;text-align:left;border:1px solid #ffeeba}.api-errors ul{padding-left:20px;margin:0}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee;display:flex;flex-direction:column;position:relative}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center;overflow:hidden}.product-image img{width:100%;height:100%;object-fit:cover}.product-info{padding:25px;display:flex;flex-direction:column;flex-grow:1;justify-content:space-between}.product-title{font-size:1.1em;font-weight:600;margin-bottom:12px;color:var(--text-color-dark)}.price-store-wrapper{display:flex;justify-content:space-between;align-items:center;margin-top:auto}.current-price{font-size:1.8em;font-weight:700;color:var(--accent-color)}.store-link a{font-weight:600;color:var(--primary-color);text-decoration:none}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas?</label><input type="text" id="query" name="query" placeholder="Ej: cinta de pintor azul 2 pulgadas"></div><div class="input-group"><label for="image_file">... o sube una imagen para una búsqueda más precisa</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Buscando las mejores ofertas en EE. UU...</p></div><section id="results-section" class="results-section"><div id="api-errors" class="api-errors" style="display:none;"></div><h2 id="results-title">Las Mejores Ofertas Encontradas</h2><div id="products-grid" class="products-grid"></div></section></div>
<script>
    const searchForm = document.getElementById("search-form");
    const queryInput = document.getElementById("query");
    const imageInput = document.getElementById("image_file");
    const loadingDiv = document.getElementById("loading");
    const resultsSection = document.getElementById("results-section");
    const productsGrid = document.getElementById("products-grid");
    const apiErrorsDiv = document.getElementById("api-errors");
    const resultsTitle = document.getElementById("results-title");

    function performSearch() {
        const formData = new FormData(searchForm);
        loadingDiv.style.display = "block";
        resultsSection.style.display = "none";
        productsGrid.innerHTML = "";
        apiErrorsDiv.innerHTML = "";
        apiErrorsDiv.style.display = "none";

        fetch("{{ url_for('api_search') }}", {
            method: "POST",
            body: formData
        }).then(response => response.json()).then(data => {
            loadingDiv.style.display = "none";

            // Lógica para mostrar mensaje de búsqueda flexible
            let isAlternative = data.results.length > 0 && data.results[0].is_alternative_suggestion;
            if (data.errors && data.errors.length > 0) {
                let errorHTML = '<ul>';
                data.errors.forEach(error => { errorHTML += `<li><strong>${error}</strong></li>`; });
                errorHTML += '</ul>';
                apiErrorsDiv.innerHTML = errorHTML;
                apiErrorsDiv.style.display = "block";
                resultsTitle.style.display = "none";
            }
            
            if (isAlternative) {
                resultsTitle.innerText = "No encontramos resultados exactos, pero aquí hay algunas opciones similares:";
            } else {
                resultsTitle.innerText = "Las Mejores Ofertas Encontradas";
            }

            if (data.results && data.results.length > 0) {
                resultsTitle.style.display = "block";
                data.results.forEach(product => {
                    const reasoning = product.reasoning.replace(/"/g, '"');
                    const originalPrice = `${product.original_price.toFixed(2)} ${product.original_currency}`;
                    productsGrid.innerHTML += `
                        <div class="product-card">
                            <div class="product-image"><img src="${product.image_url || 'https://via.placeholder.com/300'}" alt="${product.name}" onerror="this.onerror=null;this.src='https://via.placeholder.com/300';"></div>
                            <div class="product-info">
                                <div class="product-title" title="IA Reasoning: ${reasoning}">${product.name}</div>
                                <div class="price-store-wrapper">
                                    <div class="current-price" title="Original: ${originalPrice}">$${product.price_in_usd.toFixed(2)}</div>
                                    <div class="store-link"><a href="${product.url}" target="_blank">Ver en ${product.store}</a></div>
                                </div>
                            </div>
                        </div>`;
                });
            } else {
                resultsTitle.style.display = "none";
                if (!apiErrorsDiv.innerHTML) {
                    productsGrid.innerHTML = "<p>No se encontraron ofertas de alta calidad para tu búsqueda.</p>";
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
    searchForm.addEventListener("submit", function(e) { e.preventDefault(); performSearch(); });
    imageInput.addEventListener("change", function() { if (this.files && this.files[0]) { var reader = new FileReader(); reader.onload = function(e) { document.getElementById("image-preview").src = e.target.result; document.getElementById("image-preview-container").style.display = "flex"; }; reader.readAsDataURL(this.files[0]); } });
    document.getElementById("remove-image-btn").addEventListener("click", function() { imageInput.value = ""; document.getElementById("image-preview").src = "#"; document.getElementById("image-preview-container").style.display = "none"; });
</script>
</body></html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
