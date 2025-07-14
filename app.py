# app.py (versión 13.0 - Análisis de Imagen por Experto IA)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 13.0 (Expert Image Analysis with Gemini Vision)
# Novedades:
# - Se rediseña el análisis de imagen: Gemini Vision ahora describe la imagen directamente.
# - Se elimina la dependencia de google-cloud-vision, simplificando el código.
# - La IA actúa como un "experto dual" para identificar tanto piezas como tecnología.
# - El cerebro adaptativo sigue funcionando, pero ahora con consultas de imagen de alta calidad.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
import time
import statistics
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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (ADAPTATIVA Y MEJORADA)
# ==============================================================================

def _deep_scrape_content(url: str) -> Dict[str, Any]:
    headers = {'User-Agent': UserAgent().random, 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/'}
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        price_text = "N/A"
        price_selectors = ['[class*="price"]', '[id*="price"]', '[class*="Price"]', '[id*="Price"]']
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

def _get_product_category(query: str) -> str:
    if not genai: return "consumer_tech"
    print(f"  Clasificando consulta: '{query}'")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (f"Classify the following product search query. Is it for 'industrial_parts' (machinery, car parts, tools, components) or 'consumer_tech' (phones, laptops, electronics, gadgets)? "
                  f"Query: '{query}'. Respond ONLY with 'industrial_parts' or 'consumer_tech'.")
        response = model.generate_content(prompt)
        category = response.text.strip()
        print(f"  Categoría detectada: {category}")
        return category if category in ["industrial_parts", "consumer_tech"] else "consumer_tech"
    except Exception:
        return "consumer_tech"

def _verify_is_product_page(query: str, page_title: str, page_content: str, category: str) -> bool:
    if not genai: return True
    prompt_template = (f"You are a product verification analyst. The user is searching for '{query}'. I found a webpage titled '{page_title}'. Is this page offering the main product itself for sale, and not just an accessory, review, or informational article? Answer with only YES or NO.")
    print(f"  Verificando con Gemini ({category}): ¿Es '{page_title[:30]}...' una página de producto?")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt_template)
        answer = response.text.strip().upper()
        print(f"  Respuesta de verificación: {answer}")
        return answer == "YES"
    except Exception as e:
        print(f"  Error en Gemini (verificación): {e}"); return False

def _get_suggestions_with_gemini(query: str) -> List[str]:
    if not genai: return []
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"A user searched for '{query}' and found no results. Provide 3 alternative, more effective search queries. Respond with a JSON list of strings, like [\"query 1\", \"query 2\", \"query 3\"]."
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception: return []

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

    # GÉNESIS: Nueva función que usa Gemini Vision para describir la imagen.
    def get_descriptive_query_from_image(self, image_content: bytes) -> Optional[str]:
        if not genai:
            print("  ❌ Análisis con Gemini Vision saltado: Modelo no configurado.")
            return None
        print("  🧠 Analizando imagen con Gemini Vision (Modo Experto Dual)...")
        try:
            image_pil = Image.open(io.BytesIO(image_content))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = """You are an expert in identifying both industrial/automotive parts and consumer technology products.
            Analyze the following image in detail. Identify the main object, its likely material, color, potential brand, and any unique features.
            Based on your analysis, generate a single, highly effective, and specific search query in English to find this product for sale online.
            For industrial parts, be very specific about the type of part (e.g., 'engine oil pan', 'brake caliper').
            For consumer tech, include the model name if recognizable.
            Respond ONLY with the search query itself, nothing else."""
            response = model.generate_content([prompt, image_pil])
            query = response.text.strip().replace("*", "")
            print(f"  ✅ Consulta experta generada por Gemini Vision: '{query}'")
            return query
        except Exception as e:
            print(f"  ❌ Fallo CRÍTICO en análisis con Gemini Vision: {e}")
            return None
            
    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        if not genai: return f"{text_query} {image_query}"
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"Combine these into a single, effective search query. User's text: '{text_query}'. Description from image: '{image_query}'. Respond only with the final search query."
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception: return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str]]:
        text_query = query.strip() if query else None
        image_query = self.get_descriptive_query_from_image(image_content) if image_content else None
        final_query = None
        if text_query and image_query:
            final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query: final_query = text_query
        elif image_query: final_query = image_query
        if not final_query: print("❌ No se pudo determinar una consulta válida."); return [], []
        
        category = _get_product_category(final_query)
        
        print(f"🔍 Lanzando búsqueda ({category}) para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query, category)
        
        suggestions = []
        if not best_deals:
            print("🤔 No se encontraron resultados. Generando sugerencias..."); suggestions = _get_suggestions_with_gemini(final_query)
        return best_deals, suggestions

    def search_with_ai_verification(self, query: str, category: str) -> List[ProductResult]:
        if category == "industrial_parts":
            search_query = f'{query} supplier distributor'
            blacklist = ['amazon.com', 'walmart.com', 'ebay.com', 'alibaba.com', 'aliexpress.com', 'etsy.com', 'pinterest.com']
            tbm_param = None
        else:
            search_query = query
            blacklist = []
            tbm_param = "shop" # Usar la pestaña de Google Shopping para productos de consumo

        print(f"--- Búsqueda final en SerpApi: '{search_query}' ---")
        params = {"q": search_query, "engine": "google", "tbm": tbm_param, "location": "United States", "gl": "us", "hl": "en", "num": "25", "api_key": self.serpapi_key}
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            
            results_key = 'shopping_results' if category == 'consumer_tech' and 'shopping_results' in response.json() else 'organic_results'
            initial_results = response.json().get(results_key, [])

            if blacklist:
                filtered_results = [item for item in initial_results if not any(site in item.get('link', '') for site in blacklist)]
            else:
                filtered_results = initial_results

            valid_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_item = {executor.submit(_deep_scrape_content, item.get('link')): item for item in filtered_results if item.get('link')}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    content = future.result()
                    if content and content['price'] != "N/A":
                        if _verify_is_product_page(query, content['title'], content['text'], category):
                            try:
                                price_float = float(content['price'])
                                valid_results.append({'store': _get_clean_company_name(item), 'product_name': item.get('title', 'Sin título'), 'price_float': price_float, 'url': item.get('link'), 'image_url': content['image'] or item.get('thumbnail', '')})
                            except (ValueError, TypeError): continue
            
            if not valid_results: return []
            
            if len(valid_results) >= 2:
                prices = [r['price_float'] for r in valid_results]
                mean_price = statistics.mean(prices)
                price_threshold = max(0.50, mean_price / 10)
                valid_results = [r for r in valid_results if r['price_float'] >= price_threshold]

            valid_results.sort(key=lambda x: x['price_float'])
            final_results_obj = [ProductResult(name=res['product_name'], price=res['price_float'], store=res['store'], url=res['url'], image_url=res.get('image_url', '')) for res in valid_results]
            return final_results_obj[:30]
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda avanzada: {e}"); return []

# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

# ... (El resto del código de rutas y plantillas es idéntico al de la versión 9.1) ...
# (Pega aquí las rutas y las plantillas de la versión anterior)
