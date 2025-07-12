# app.py (versión 8.0 - Descripción de Imágenes con Gemini Vision)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 8.0 (Image-to-Text with Gemini Vision)
# Novedades:
# - Se reemplaza OpenAI CLIP por Gemini Vision para el análisis de imágenes.
# - La IA ahora "describe" la imagen en detalle para generar una consulta de búsqueda precisa.
# - Se eliminan las librerías pesadas (torch, clip), mejorando drásticamente la estabilidad y el uso de RAM.
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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (CON GEMINI VISION)
# ==============================================================================

# ... (Las funciones _deep_scrape_content, _get_relevance_score_with_gemini, etc. se quedan igual) ...

@dataclass
class ProductResult:
    name: str; price: float; store: str; url: str; image_url: str = ""; relevance_score: int = 0

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key

    # GÉNESIS: Nueva función que usa Gemini Vision para describir la imagen.
    def get_descriptive_query_with_gemini_vision(self, image_content: bytes) -> Optional[str]:
        """Usa Gemini Vision para describir una imagen y generar una consulta de búsqueda."""
        if not genai:
            print("  ❌ Análisis con Gemini Vision saltado: Modelo no configurado.")
            return None
        
        print("  🧠 Analizando imagen con Gemini Vision...")
        try:
            image_pil = Image.open(io.BytesIO(image_content))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            prompt = """
            You are an expert in identifying products and parts from images.
            Analyze the following image in detail. Identify the main object, its likely material, color, and any unique features.
            Based on your analysis, generate a single, highly effective, and specific search query in English to find this product for sale online.
            Respond ONLY with the search query itself, nothing else.
            """
            
            response = model.generate_content([prompt, image_pil])
            
            # Limpiamos la respuesta para quedarnos solo con la consulta
            query = response.text.strip()
            print(f"  ✅ Consulta generada por Gemini Vision: '{query}'")
            return query
        except Exception as e:
            print(f"  ❌ Fallo en análisis con Gemini Vision: {e}")
            return None

    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        # ... (la lógica de combinación se queda igual, pero ahora recibe una descripción, no keywords) ...
        if not genai: return f"{text_query} {image_query}"
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"A user is searching for a product. User's text: '{text_query}'. Description from image: '{image_query}'. Combine these into a single, effective search query. Prioritize details from the user's text. Respond only with the final search query."
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str]]:
        text_query = query.strip() if query else None
        # GÉNESIS: Se llama a la nueva función de Gemini Vision
        image_query = self.get_descriptive_query_with_gemini_vision(image_content) if image_content else None
        
        final_query = None
        if text_query and image_query:
            print(f"🧠 Combinando texto '{text_query}' e imagen (descripción IA: '{image_query}')...")
            final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query:
            final_query = text_query
        elif image_query:
            final_query = image_query

        if not final_query:
            print("❌ No se pudo determinar una consulta válida."); return [], []
        
        print(f"🔍 Lanzando búsqueda neuronal para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query)
        
        suggestions = []
        if not best_deals:
            print("🤔 No se encontraron resultados. Generando sugerencias...")
            suggestions = _get_suggestions_with_gemini(final_query)
        return best_deals, suggestions

    def search_with_ai_verification(self, search_query: str) -> List[ProductResult]:
        # GÉNESIS: Usamos la versión optimizada de la v6.2
        params = {"q": search_query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "20", "api_key": self.serpapi_key}
        try:
            # ... (el resto de esta función es idéntico a la versión anterior) ...
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            organic_results = response.json().get('organic_results', [])
            results_with_scores = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_item = {executor.submit(_deep_scrape_content, item.get('link')): item for item in organic_results if item.get('link')}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    content = future.result()
                    if content and content['price'] != "N/A":
                        try:
                            price_float = float(content['price'])
                            relevance_score = _get_relevance_score_with_gemini(search_query, content['title'], content['text'])
                            if relevance_score >= 5:
                                results_with_scores.append({'store': _get_clean_company_name(item), 'product_name': item.get('title', 'Sin título'), 'price_float': price_float, 'url': item.get('link'), 'image_url': content['image'] or item.get('thumbnail', ''), 'relevance_score': relevance_score})
                        except (ValueError, TypeError): continue
            if not results_with_scores: return []
            results_with_scores.sort(key=lambda x: (-x['relevance_score'], x['price_float']))
            final_results_obj = [ProductResult(name=res['product_name'], price=res['price_float'], store=res['store'], url=res['url'], image_url=res.get('image_url', ''), relevance_score=res['relevance_score']) for res in results_with_scores]
            return final_results_obj[:30]
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda avanzada: {e}"); return []


# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

# ... (las rutas de Flask y las plantillas HTML se quedan exactamente igual que en la v6.2) ...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
