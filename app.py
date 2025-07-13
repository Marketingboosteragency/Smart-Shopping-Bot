# app.py (versión 9.0 - Verificación Semántica de Productos)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 9.0 (Semantic Product Page Verification)
# Novedades:
# - Se integra la lógica de "descartar por inutilidad", usando IA para verificar si una
#   página vende el producto o es solo un artículo/accesorio.
# - Se reintroduce y mejora el análisis de imagen: Google Vision extrae pistas y
#   Gemini las sintetiza en una consulta de búsqueda óptima.
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
    from google.cloud import vision
    print("✅ Módulo de Google Cloud Vision importado.")
except ImportError:
    print("⚠️ AVISO: 'google-cloud-vision' no está instalado. El análisis de imagen no funcionará.")
    vision = None
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
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get('GOOGLE_CREDENTIALS_JSON') # Necesario para Vision
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-muy-fuerte')

# Configuración de Gemini
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR al configurar API de Gemini: {e}")
        genai = None

# Configuración de Google Cloud Vision
if GOOGLE_CREDENTIALS_JSON_STR and vision:
    try:
        google_creds_info = json.loads(GOOGLE_CREDENTIALS_JSON_STR)
        with open('/tmp/google-credentials.json', 'w') as f:
            json.dump(google_creds_info, f)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/google-credentials.json'
        print("✅ Credenciales de Google Vision cargadas.")
    except Exception as e:
        print(f"❌ ERROR al cargar credenciales de Google Vision: {e}")

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (CON VERIFICACIÓN SEMÁNTICA)
# ==============================================================================

# ... (las funciones _deep_scrape_content y _get_clean_company_name se quedan igual) ...

# GÉNESIS: Nueva función de verificación de página de producto
def _verify_is_product_page(original_query: str, page_title: str, page_content: str) -> bool:
    """Usa a Gemini para verificar si la página es relevante para comprar el producto."""
    if not genai: return True # Asumir que es válido si no hay IA
    
    print(f"  Verificando con Gemini: ¿Es '{page_title[:30]}...' una página de producto para '{original_query}'?")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (
            f"You are a product verification analyst. The user is searching for '{original_query}'. "
            f"I found a webpage with the title '{page_title}'. "
            f"Here is some text from that page: '{page_content[:500]}'. "
            "Is this page offering the main product itself for sale, and not just an accessory, a part, or an informational article? "
            "Answer with only the word YES or NO."
        )
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        print(f"  Respuesta de verificación de Gemini: {answer}")
        return answer == "YES"
    except Exception as e:
        print(f"  Error en Gemini (verificación): {e}")
        return False # Si la IA falla, es más seguro descartar

@dataclass
class ProductResult:
    name: str; price: float; store: str; url: str; image_url: str = ""

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.vision_client = None
        if vision and GOOGLE_CREDENTIALS_JSON_STR:
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                print("✅ Cliente de Google Cloud Vision inicializado.")
            except Exception as e:
                print(f"❌ ERROR CRÍTICO EN VISION INIT: {e}")

    # GÉNESIS: Nueva lógica de análisis de imagen que agrega pistas y sintetiza una consulta
    def _aggregate_vision_results(self, response):
        clues = []
        if response.web_detection and response.web_detection.best_guess_labels:
            clues.append(f"Best Guess from web: {response.web_detection.best_guess_labels[0].label}")
        if response.logo_annotations:
            clues.append(f"Logos Detected: {', '.join([logo.description for logo in response.logo_annotations])}")
        if response.label_annotations:
            clues.append(f"Labels: {', '.join([label.description for label in response.label_annotations[:3]])}")
        return ". ".join(clues)

    def get_query_from_image(self, image_content: bytes) -> Optional[str]:
        if not self.vision_client:
            print("  ❌ Análisis con Vision saltado: Cliente no inicializado.")
            return None
        
        print("  🧠 Analizando imagen con Google Cloud Vision (Multi-Feature)...")
        try:
            image_for_api = vision.Image(content=image_content)
            features = [
                vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
            ]
            request_body = vision.AnnotateImageRequest(image=image_for_api, features=features)
            response = self.vision_client.annotate_image(request=request_body)
            
            aggregated_clues = self._aggregate_vision_results(response)
            
            if not aggregated_clues or not genai:
                return response.web_detection.best_guess_labels[0].label if response.web_detection else None

            print(f"  Synthesizing search term from clues: '{aggregated_clues}'")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"You are a search query synthesizer. Based on these data points from an image analysis, create the best possible search query in English. DATA: '{aggregated_clues}'. Respond ONLY with the synthesized search query."
            gemini_response = model.generate_content(prompt)
            search_term = gemini_response.text.strip().replace('\n', '')
            print(f"  ✅ Consulta sintetizada por Gemini: '{search_term}'")
            return search_term
        except Exception as e:
            print(f"  ❌ Fallo en análisis de imagen o síntesis: {e}")
            return None

    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        # ... (esta función se queda igual)
        if not genai: return f"{text_query} {image_query}"
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"Combine these into a single, effective search query. User's text: '{text_query}'. Description from image: '{image_query}'. Respond only with the final search query."
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str]]:
        text_query = query.strip() if query else None
        image_query = self.get_query_from_image(image_content) if image_content else None
        
        final_query = None
        if text_query and image_query:
            print(f"🧠 Combinando texto '{text_query}' e imagen (descripción IA: '{image_query}')...")
            final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query: final_query = text_query
        elif image_query: final_query = image_query
        if not final_query: print("❌ No se pudo determinar una consulta válida."); return [], []
        
        print(f"🔍 Lanzando búsqueda neuronal para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query)
        
        suggestions = []
        if not best_deals:
            print("🤔 No se encontraron resultados. Generando sugerencias...")
            suggestions = _get_suggestions_with_gemini(final_query)
        return best_deals, suggestions

    def search_with_ai_verification(self, search_query: str) -> List[ProductResult]:
        params = {"q": search_query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "20", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            organic_results = response.json().get('organic_results', [])
            
            valid_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_item = {executor.submit(_deep_scrape_content, item.get('link')): item for item in organic_results if item.get('link')}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    content = future.result()
                    if content and content['price'] != "N/A":
                        # GÉNESIS: Se llama a la nueva función de verificación
                        if _verify_is_product_page(search_query, content['title'], content['text']):
                            try:
                                price_float = float(content['price'])
                                valid_results.append({'store': _get_clean_company_name(item), 'product_name': item.get('title', 'Sin título'), 'price_float': price_float, 'url': item.get('link'), 'image_url': content['image'] or item.get('thumbnail', '')})
                            except (ValueError, TypeError): continue
            
            if not valid_results: return []

            # Ordenar por precio
            valid_results.sort(key=lambda x: x['price_float'])
            
            final_results_obj = [ProductResult(name=res['product_name'], price=res['price_float'], store=res['store'], url=res['url'], image_url=res.get('image_url', '')) for res in valid_results]
            return final_results_obj[:30]
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda avanzada: {e}"); return []

# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
# ... (el resto del código, incluyendo rutas y plantillas, no necesita cambios) ...
