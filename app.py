# app.py (versión 10.0 - Búsqueda Genérica y Filtro Semántico)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 10.0 (Generic Search & Semantic Filtering)
# Novedades:
# - Se integra la lógica de "descartar por inutilidad" para filtrar resultados.
# - Se añade "in USA" a las consultas para mejorar la búsqueda de productos genéricos.
# - Se limpia el código para eliminar el antiguo sistema de puntaje de relevancia.
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
    print("⚠️ AVISO: 'google-cloud-vision' no está instalado.")
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
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get('GOOGLE_CREDENTIALS_JSON')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-muy-fuerte')

# Configuración de Gemini y Google Cloud Vision
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR al configurar API de Gemini: {e}")
        genai = None

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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (MEJORADA)
# ==============================================================================

# ... (las funciones _deep_scrape_content, _get_suggestions_with_gemini, _get_clean_company_name se quedan igual) ...

# GÉNESIS: Integrada tu función de verificación de página de producto
def _verify_is_product_page(original_query: str, page_title: str, page_content: str) -> bool:
    if not genai: return True
    print(f"  Verificando con Gemini: ¿Es '{page_title[:30]}...' una página de producto para '{original_query}'?")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (f"You are a product verification analyst. The user is searching for '{original_query}'. "
                  f"I found a webpage with the title '{page_title}'. "
                  f"Here is some text from that page: '{page_content[:500]}'. "
                  "Is this page offering the main product itself for sale, and not just an accessory, a part, or an informational article? "
                  "Answer with only the word YES or NO.")
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        print(f"  Respuesta de verificación de Gemini: {answer}")
        return answer == "YES"
    except Exception as e:
        print(f"  Error en Gemini (verificación): {e}"); return False

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

    # ... (las funciones _aggregate_vision_results, get_query_from_image, _combine_text_and_image_query se quedan igual) ...

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str]]:
        text_query = query.strip() if query else None
        image_query = self.get_query_from_image(image_content) if image_content else None
        
        is_text_search = bool(text_query and not image_content)
        
        final_query = None
        if text_query and image_query:
            print(f"🧠 Combinando texto '{text_query}' e imagen (descripción IA: '{image_query}')...")
            final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query: final_query = text_query
        elif image_query: final_query = image_query
        if not final_query: print("❌ No se pudo determinar una consulta válida."); return [], []
        
        print(f"🔍 Lanzando búsqueda neuronal para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query, is_text_search)
        
        suggestions = []
        if not best_deals:
            print("🤔 No se encontraron resultados. Generando sugerencias...")
            suggestions = _get_suggestions_with_gemini(final_query)
        return best_deals, suggestions

    def search_with_ai_verification(self, query: str, is_text_search: bool = False) -> List[ProductResult]:
        # GÉNESIS: Se añade "in USA" o "price in USA" a la consulta para mejorar resultados genéricos
        search_query = f'{query} in USA' if is_text_search else f'{query} price in USA'
        print(f"--- Búsqueda final en SerpApi: '{search_query}' ---")
        
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
                        # GÉNESIS: Se usa la nueva función de verificación
                        if _verify_is_product_page(query, content['title'], content['text']):
                            try:
                                price_float = float(content['price'])
                                valid_results.append({'store': _get_clean_company_name(item), 'product_name': item.get('title', 'Sin título'), 'price_float': price_float, 'url': item.get('link'), 'image_url': content['image'] or item.get('thumbnail', '')})
                            except (ValueError, TypeError): continue
            
            if not valid_results: return []
            
            # GÉNESIS: Lógica de precios para descartar outliers (del script de Firebase)
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

# ... (El resto del código, incluyendo las rutas y plantillas, no sufre cambios lógicos, pero la plantilla SEARCH_TEMPLATE necesita un pequeño ajuste) ...

# ==============================================================================
# SECCIÓN 4: PLANTILLAS HTML Y EJECUCIÓN
# ==============================================================================
AUTH_TEMPLATE_LOGIN_ONLY = """ (Tu plantilla de Login aquí) """
# GÉNESIS: Plantilla de búsqueda actualizada para eliminar la puntuación de relevancia
SEARCH_TEMPLATE = """
<!DOCTYPE html><html lang="es"><head> ... </head><body>
... (contenido del header y search-section) ...
<section id="results-section" class="results-section">
    <h2 id="results-title">Mejores Ofertas Encontradas</h2>
    <div id="suggestions"></div>
    <div id="products-grid" class="products-grid"></div>
</section>
... (resto del body) ...
<script>
// ... (script sin cambios lógicos, pero la parte que genera las tarjetas de producto se actualiza)
function performSearch() {
    // ...
    fetch("{{ url_for('api_search') }}", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => {
        // ...
        if (data.results && data.results.length > 0) {
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
        }
        // ...
    });
}
// ...
</script>
</body></html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
