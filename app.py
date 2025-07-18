# app.py (versión 15.2 - Motor de Autoridad de Precios - Single File)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 15.2 (Price Authority Engine - Single File Architecture)
# Novedades:
# - ARQUITECTURA DE ARCHIVO ÚNICO: Todo el código, incluyendo el HTML, está contenido en este único archivo para un despliegue simple, corrigiendo el `SyntaxError` anterior.
# - MANTIENE EL MOTOR DE AUTORIDAD DE PRECIOS: La IA sigue siendo la única responsable de extraer precios y divisas para máxima precisión.
# - MANTIENE LA CONVERSIÓN DE DIVISAS: Todos los precios se normalizan a USD para una comparación justa.
# - MANTIENE LA VALIDACIÓN HOLÍSTICA: La IA juzga la relevancia y la confianza de cada resultado.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
import io
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
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
    name: str
    price_in_usd: float
    store: str
    url: str
    image_url: str = ""
    original_price: float = 0.0
    original_currency: str = "USD"
    is_validated: bool = False
    relevance_reasoning: str = ""
    confidence_score: float = 0.0

# Tasas de cambio aproximadas a USD. En una app de producción, esto vendría de una API.
CURRENCY_RATES_TO_USD = {
    "USD": 1.0, "DOP": 0.017, "MXN": 0.054, "CAD": 0.73, "EUR": 1.08, "GBP": 1.27
}

def _deep_scrape_content(url: str) -> Dict[str, Any]:
    """Scraper optimizado para recolectar la máxima cantidad de texto contextual."""
    headers = {'User-Agent': UserAgent().random, 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/'}
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        image_url = (og_image.get("content") for og_image in [soup.find("meta", property="og:image")] if og_image)
        image_url = urljoin(url, next(image_url, ''))
        
        title = soup.title.string.strip() if soup.title else 'No Title'
        text_content = ' '.join(soup.stripped_strings)[:1500] # Aumentamos la captura de texto
        
        return {'title': title, 'image': image_url, 'text_content': text_content, 'url': url}
    except Exception:
        return {'title': 'N/A', 'image': '', 'text_content': '', 'url': url}

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

def _get_price_and_relevance_from_ai(candidate: Dict[str, Any], original_query: str, errors_list: List[str]) -> Dict[str, Any]:
    """La IA actúa como la autoridad final para extraer precio, moneda y relevancia."""
    default_failure = {"is_highly_relevant": False, "confidence_score": 0.0}
    if not genai or not candidate.get('text_content'): return default_failure
    
    print(f"  🤖⚖️ Sometiendo a juicio de IA: '{candidate['title']}'...")
    
    prompt = (
        f"You are a precise data extraction AI for e-commerce. Analyze the following data and return a JSON object.\n\n"
        f"DATA:\n"
        f"- User's Search: '{original_query}'\n"
        f"- Page Title: '{candidate['title']}'\n"
        f"- Page Text Snippet: '{candidate['text_content']}'\n\n"
        f"TASKS:\n"
        f"1.  **Extract Price & Currency:** Find the main product's price in the text. Identify its 3-letter currency code (e.g., 'USD', 'MXN', 'DOP'). If no currency is obvious, assume 'USD'.\n"
        f"2.  **Check Relevance:** Is the product on the page a direct, relevant answer to the user's search?\n\n"
        f"Provide your analysis in a JSON object with these exact keys:\n"
        f"- `is_highly_relevant`: boolean (true only if it directly matches the user's search).\n"
        f"- `price`: float (the extracted numerical price. If a price range, use the lower value).\n"
        f"- `currency`: string (the 3-letter ISO 4217 currency code).\n"
        f"- `confidence_score`: float (from 0.0 to 1.0, your confidence in this entire analysis).\n"
        f"- `reasoning`: string (a brief explanation)."
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        analysis = json.loads(cleaned_response)
        
        required_keys = ["is_highly_relevant", "price", "currency", "confidence_score", "reasoning"]
        if not all(key in analysis for key in required_keys): return default_failure
        
        print(f"  🧠 Juicio de IA completado: Relevante={analysis['is_highly_relevant']}, Precio={analysis['price']} {analysis['currency']}, Confianza={analysis['confidence_score']:.2f}")
        return analysis
    except (json.JSONDecodeError, google_exceptions.ResourceExhausted) as e:
        if isinstance(e, google_exceptions.ResourceExhausted):
            error_msg = "Advertencia: Cuota de API superada durante el juicio de IA. La calidad de los resultados se verá gravemente afectada."
            if error_msg not in errors_list: errors_list.append(error_msg)
        return default_failure
    except Exception:
        return default_failure

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.CONFIDENCE_THRESHOLD = 0.75

    def _get_candidate_urls(self, query: str) -> List[str]:
        print(f"--- Recolectando URLs para: '{query}' ---")
        urls = set()
        for engine in ["google", "google_shopping"]:
            params = {"q": query, "engine": engine, "location": "United States", "gl": "us", "hl": "en", "api_key": self.serpapi_key}
            if engine == "google": params["num"] = "15"
            try:
                response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
                response.raise_for_status()
                data = response.json()
                results = data.get('organic_results', []) if engine == "google" else data.get('shopping_results', [])
                for item in results:
                    if isinstance(item, dict) and item.get('link'):
                        urls.add(item['link'])
            except Exception as e:
                print(f"❌ Ocurrió un error en la recolección de URLs ({engine}): {e}")
        return list(urls)

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str], List[str]]:
        errors_list = []
        original_query = query.strip() if query else "product from image"
        if not original_query: return [], ["Por favor, introduce un texto o sube una imagen."], []

        enhanced_query = _enhance_query_for_purchase(original_query, errors_list)
        if not enhanced_query: return [], ["No se pudo generar una consulta válida."], errors_list
        
        print(f"--- FASE 1: Recolectando candidatos para '{enhanced_query}' ---")
        candidate_urls = self._get_candidate_urls(enhanced_query)
        blacklist = ['pinterest.com', 'youtube.com', 'wikipedia.org', 'facebook.com', 'twitter.com', 'yelp.com']
        filtered_urls = [url for url in candidate_urls if not any(site in url for site in blacklist)]
        
        print(f"--- {len(filtered_urls)} URLs candidatas pasarán a la fase de scrape y juicio. ---")
        if not filtered_urls: return [], [], errors_list

        validated_products = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {executor.submit(_deep_scrape_content, url): url for url in filtered_urls}
            scraped_candidates = [future.result() for future in as_completed(future_to_url)]
            
            candidates_for_judgement = [c for c in scraped_candidates if c['text_content']]
            print(f"--- FASE 2: Sometiendo a juicio de IA a {len(candidates_for_judgement)} candidatos con contenido. ---")
            
            future_to_candidate = {executor.submit(_get_price_and_relevance_from_ai, c, original_query, errors_list): c for c in candidates_for_judgement}
            for future in as_completed(future_to_candidate):
                candidate_data = future_to_candidate[future]
                try:
                    analysis = future.result()
                    if analysis['is_highly_relevant'] and analysis.get('confidence_score', 0) >= self.CONFIDENCE_THRESHOLD:
                        currency = analysis.get('currency', 'USD').upper()
                        rate = CURRENCY_RATES_TO_USD.get(currency)
                        if rate:
                            original_price = float(analysis['price'])
                            price_in_usd = original_price * rate
                            
                            if price_in_usd >= 0.50:
                                validated_products.append(ProductResult(
                                    name=candidate_data['title'],
                                    price_in_usd=price_in_usd,
                                    store=urlparse(candidate_data['url']).netloc.replace('www.', '').split('.')[0].capitalize(),
                                    url=candidate_data['url'],
                                    image_url=candidate_data['image'],
                                    original_price=original_price,
                                    original_currency=currency,
                                    relevance_reasoning=analysis['reasoning'],
                                    confidence_score=analysis['confidence_score'],
                                    is_validated=True
                                ))
                except Exception as e:
                    print(f"  ❌ Error procesando el juicio del producto {candidate_data['title']}: {e}")
        
        if not validated_products:
            print("🤔 Después del juicio de IA, no quedaron resultados de alta calidad.")
            return [], [], errors_list
            
        final_results = sorted(validated_products, key=lambda x: x.price_in_usd)
        print(f"✅ BÚSQUEDA COMPLETA. Se encontraron {len(final_results)} resultados de alta calidad verificados por IA.")
        return final_results, [], errors_list

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
    if not FIREBASE_WEB_API_KEY:
        flash('El servicio de autenticación no está configurado.', 'danger')
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
        error_message = e.response.json().get('error', {}).get('message', 'ERROR_DESCONOCIDO')
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
    return render_template_string(SEARCH_TEMPLATE, user_name=session.get('user_name', 'Usuario'))

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
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Acceso | Smart Shopping Bot</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--text-color-dark:#2C3E50;--card-bg:#FFFFFF;--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:linear-gradient(135deg,var(--primary-color) 0%,var(--secondary-color) 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}.auth-container{max-width:480px;width:100%;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-medium);overflow:hidden;animation:fadeIn .8s ease-out}@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.form-header{text-align:center;padding:40px 30px 20px}.form-header h1{color:var(--text-color-dark);font-size:2em;margin-bottom:10px}.form-header p{color:#7f8c8d;font-size:1.1em}.form-body{padding:10px 40px 40px
