# app.py (versión 26.0 - Plantillas Externalizadas)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 26.0 (Externalized Templates)
# Novedades:
# - CÓDIGO LIMPIO: Las plantillas HTML se han movido a la carpeta /templates.
# - MEJORES PRÁCTICAS: Se utiliza render_template en lugar de render_template_string para una estructura de proyecto profesional y fácil de mantener.
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
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash # ¡CAMBIO! Se usa render_template
from PIL import Image

# --- IMPORTS DE APIS DE GOOGLE ---
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    print("✅ Módulo de Google Generative AI (Gemini) importado.")
except ImportError:
    print("⚠️ AVISO: 'google-generativeai' no está instalado.")
    genai = None; google_exceptions = None

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    print("✅ Módulo de Google API Client (para Búsqueda) importado.")
except ImportError:
    print("⚠️ AVISO: 'google-api-python-client' no está instalado.")
    build = None; HttpError = None

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL DE FLASK Y APIS
# ==============================================================================
app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PROGRAMMABLE_SEARCH_ENGINE_ID = os.environ.get("PROGRAMMABLE_SEARCH_ENGINE_ID")
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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (sin cambios)
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
        f"You are a critical shopping expert AI. Your goal is to find the best value. Analyze this product page data and return a JSON object.\n\n"
        f"DATA:\n"
        f"- User's Search: '{original_query}'\n"
        f"- Page Title: '{candidate['title']}'\n"
        f"- Page Text (first 2000 chars): '{candidate['text_content']}'\n\n"
        f"TASKS & SCORING:\n"
        f"1.  **Extract Price & Currency:** Find the *final, non-sale price* for a single unit. Use 'USD' if unclear. If you see words like 'sale', 'clearance', or 'discount', note it in your reasoning.\n"
        f"2.  **Relevance Score (1-10):** Be strict. How perfectly does this product match the user's search? 10 is an exact match. Below 6 means it's a different model, color, or a related accessory. Be very critical.\n"
        f"3.  **Price Accuracy Score (1-10):** How confident are you that this is the main product's price, not a price for a part, a subscription, or a bulk offer? 10 is very confident.\n"
        f"4.  **US Centric Check:** Does this store clearly operate in or ship to the USA? This is a critical requirement.\n\n"
        f"Return a single, valid JSON object with these keys: `price` (float), `currency` (string), `relevance_score` (int), `price_accuracy_score` (int), `is_usa_centric` (boolean), `reasoning` (string, max 20 words, explaining your relevance and price decision)."
    )
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        analysis = json.loads(response.text)
        if not all(k in analysis for k in ["price", "currency", "relevance_score", "price_accuracy_score", "is_usa_centric"]): return default_failure
        print(f"  🧠 Calificación IA: Relevancia={analysis['relevance_score']}/10, Precisión={analysis['price_accuracy_score']}/10, Precio={analysis.get('price')} {analysis.get('currency')}, USA?={analysis.get('is_usa_centric')}")
        return analysis
    except (json.JSONDecodeError, google_exceptions.ResourceExhausted, ValueError) as e:
        if isinstance(e, google_exceptions.ResourceExhausted): errors_list.append("Advertencia: Cuota de API superada durante el análisis.")
        return default_failure
    except Exception: return default_failure

class SmartShoppingBot:
    def __init__(self, google_api_key: str, search_engine_id: str):
        self.google_api_key = google_api_key
        self.search_engine_id = search_engine_id
        self.search_service = None
        if not all([google_api_key, search_engine_id, build]):
             print("❌ ERROR: Faltan las credenciales de la API de Google o la librería no está instalada.")
             return
        try:
            self.search_service = build("customsearch", "v1", developerKey=self.google_api_key)
            print("✅ Servicio de Búsqueda de Google (Custom Search) inicializado.")
        except Exception as e:
            print(f"❌ ERROR al inicializar el servicio de Búsqueda de Google: {e}")
        
        self.high_priority_stores = ["amazon.com", "walmart.com", "ebay.com", "target.com", "bestbuy.com"]
        self.home_improvement_stores = ["homedepot.com", "lowes.com", "acehardware.com", "harborfreight.com"]
        self.discount_retailers = ["overstock.com", "wayfair.com", "newegg.com", "bhphotovideo.com"]
            
        self.MAX_RESULTS_TO_RETURN = 30
        self.MINIMUM_RESULTS_TARGET = 10

    def _run_single_search_task(self, query: str, start: int = 1) -> List[str]:
        urls = []
        if not self.search_service:
            print("❌ El servicio de búsqueda de Google no está disponible.")
            return []
        try:
            print(f"  📡 Ejecutando búsqueda en Google: Query='{query[:50]}...', Página={int(start / 10) + 1}")
            result = self.search_service.cse().list(
                q=query, cx=self.search_engine_id, num=10, start=start, gl='us', hl='en'
            ).execute()
            items = result.get('items', [])
            for item in items:
                if 'link' in item: urls.append(item['link'])
        except HttpError as e:
            error_details = json.loads(e.content).get('error', {})
            error_message = error_details.get('message', 'Error desconocido de la API de Google.')
            print(f"❌ Error en la API de Búsqueda de Google: {error_message}")
        except Exception as e:
            print(f"❌ Error inesperado en sub-búsqueda (Google): {e}")
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
                
                extracted_price = analysis.get('price')
                if analysis.get('is_usa_centric', False) and analysis.get('relevance_score', 0) >= 6 and analysis.get('price_accuracy_score', 0) >= 5 and extracted_price is not None:
                    currency = analysis.get('currency', 'USD').upper(); rate = CURRENCY_RATES_TO_USD.get(currency)
                    if rate:
                        try:
                            original_price = float(extracted_price)
                            price_in_usd = original_price * rate
                            if price_in_usd >= 0.50:
                                analyzed_products.append(ProductResult(
                                    name=candidate_data['title'], store=urlparse(candidate_data['url']).netloc.replace('www.', '').split('.')[0].capitalize(),
                                    url=candidate_data['url'], image_url=candidate_data['image'],
                                    price_in_usd=price_in_usd, original_price=original_price, original_currency=currency,
                                    relevance_score=analysis['relevance_score'], price_accuracy_score=analysis['price_accuracy_score'], 
                                    reasoning=analysis.get('reasoning', ''), is_alternative_suggestion=is_fallback
                                ))
                        except (ValueError, TypeError):
                            print(f"  ⚠️  Precio inválido '{extracted_price}' para '{candidate_data['title']}'. Omitiendo.")
                            pass

        return analyzed_products

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str], List[str]]:
        errors_list = []
        try:
            original_query = query.strip() if query else "product from image"
            if not original_query: return [], [], []

            enhanced_query = _enhance_query_for_purchase(original_query, errors_list)
            if not enhanced_query: return [], ["No se pudo generar una consulta válida."], errors_list
            
            all_urls = set()
            
            print("--- OLEADA 1: Búsqueda de Alta Precisión ---")
            queries_wave_1 = [ f'"{enhanced_query}" buy online usa', f'{enhanced_query} price' ]
            for q in queries_wave_1:
                for url in self._run_single_search_task(q, start=1): all_urls.add(url)

            print("--- OLEADA 2: Búsqueda en Tiendas Prioritarias ---")
            store_query_part = " OR ".join([f"site:{store}" for store in self.high_priority_stores])
            store_query = f'({store_query_part}) "{enhanced_query}"'
            for url in self._run_single_search_task(store_query, start=1): all_urls.add(url)
            
            final_results = self._process_and_validate_candidates(list(all_urls), original_query, errors_list)

            if len(final_results) < self.MINIMUM_RESULTS_TARGET:
                print(f"--- OLEADA 3: Expandiendo a Búsqueda Amplia y de Descuento ---")
                for q in [ f'{enhanced_query} sale', f'{enhanced_query} discount' ]:
                    for url in self._run_single_search_task(q, start=1): all_urls.add(url)
                
                niche_stores = self.home_improvement_stores + self.discount_retailers
                niche_query_part = " OR ".join([f"site:{store}" for store in niche_stores])
                niche_query = f'({niche_query_part}) "{enhanced_query}"'
                for url in self._run_single_search_task(niche_query, start=1): all_urls.add(url)

                final_results = self._process_and_validate_candidates(list(all_urls), original_query, errors_list)

            if not final_results:
                print("--- OLEADA 4: Búsqueda Flexible (Fallback) ---")
                fallback_query = _get_fallback_query_from_ai(original_query, errors_list)
                if fallback_query:
                    fallback_urls = self._run_single_search_task(fallback_query, start=1)
                    final_results = self._process_and_validate_candidates(fallback_urls, original_query, errors_list, is_fallback=True)
                    if final_results:
                        errors_list.insert(0, "No encontramos resultados exactos. Pero aquí hay algunas opciones similares que podrían interesarte.")

            if not final_results: 
                print("✅ BÚSQUEDA COMPLETA. No se encontraron ofertas de alta calidad.")
                return [], [], errors_list
            
            final_results = sorted(final_results, key=lambda p: ( (p.relevance_score ** 2) / p.price_in_usd if p.price_in_usd > 0 else 0), reverse=True)
            
            print(f"✅ BÚSQUEDA COMPLETA. Se encontraron {len(final_results)} ofertas de calidad, ordenadas por el mejor puntaje.")
            return final_results[:self.MAX_RESULTS_TO_RETURN], [], errors_list

        except Exception as e:
            print(f"‼️ ERROR CRÍTICO NO MANEJADO EN search_product: {e}")
            traceback.print_exc()
            errors_list.append("Ocurrió un error inesperado en el servidor.")
            return [], [], errors_list
# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
shopping_bot = SmartShoppingBot(google_api_key=GOOGLE_API_KEY, search_engine_id=PROGRAMMABLE_SEARCH_ENGINE_ID)

@app.route('/')
def index():
    if 'user_id' in session: 
        return redirect(url_for('main_app_page'))
    # ¡CAMBIO! Llama al archivo login.html
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    if not FIREBASE_WEB_API_KEY: 
        flash('Servicio no configurado.', 'danger')
        return redirect(url_for('index'))
    email, password = request.form.get('email'), request.form.get('password')
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {'email': email, 'password': password, 'returnSecureToken': True}
    try:
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status()
        user_data = response.json()
        session['user_id'] = user_data['localId']
        session['user_name'] = user_data.get('displayName', email)
        session['id_token'] = user_data['idToken']
        flash('¡Has iniciado sesión correctamente!', 'success')
        return redirect(url_for('main_app_page'))
    except requests.exceptions.HTTPError as e:
        error_message = e.response.json().get('error', {}).get('message', 'ERROR')
        flash('Correo o contraseña incorrectos.' if 'INVALID' in error_message else f'Error: {error_message}', 'danger')
        return redirect(url_for('index'))
    except Exception as e: 
        flash(f'Ocurrió un error inesperado: {e}', 'danger')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Has cerrado la sesión.', 'success')
    return redirect(url_for('index'))

@app.route('/app')
def main_app_page():
    if 'user_id' not in session: 
        flash('Debes iniciar sesión para acceder.', 'warning')
        return redirect(url_for('index'))
    # ¡CAMBIO! Llama al archivo search.html y le pasa el nombre de usuario
    return render_template('search.html', user_name=session.get('user_name', 'Usuario'))

@app.route('/api/search', methods=['POST'])
def api_search():
    if 'user_id' not in session: 
        return jsonify({'error': 'No autorizado'}), 401
    query = request.form.get('query')
    image_file = request.files.get('image_file')
    image_content = image_file.read() if image_file and image_file.filename != '' else None
    results, _, errors = shopping_bot.search_product(query=query, image_content=image_content)
    results_dicts = [res.__dict__ for res in results]
    return jsonify(results=results_dicts, suggestions=[], errors=errors)

# ==============================================================================
# SECCIÓN 4: EJECUCIÓN (YA NO HAY PLANTILLAS AQUÍ)
# ==============================================================================
# ¡CAMBIO! Se eliminan las plantillas gigantes de aquí.

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
