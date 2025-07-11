# app.py (versión 6.0 - Búsqueda Neuronal Avanzada)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 6.0 (Neural-Powered Search & Relevance Engine)
# Novedades:
# - Búsqueda por imagen mejorada: Usa NLP para crear consultas ricas.
# - Búsqueda combinada: Fusiona texto e imagen para consultas ultra-precisas.
# - Puntuación de Relevancia: La IA asigna un puntaje de 1 a 10 a cada producto.
# - Ordenamiento avanzado: Los resultados se ordenan por relevancia y luego por precio.
# - Scraping reforzado: Lógica mejorada para extraer precios e imágenes de alta calidad.
# - Sugerencias inteligentes: Si no hay resultados, la IA sugiere búsquedas alternativas.
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
import json
import os
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash
from collections import Counter

# --- IMPORTS DE APIs DE GOOGLE ---
try:
    from google.cloud import vision
    print("✅ Módulo de Google Cloud Vision importado.")
except ImportError:
    print("❌ ERROR: 'google-cloud-vision' no está instalado.")
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
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get('GOOGLE_CREDENTIALS_JSON')
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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (MEJORADA)
# ==============================================================================

def _deep_scrape_content(url: str) -> Dict[str, Any]:
    """Scraping reforzado para extraer título, texto, precio e imagen de alta calidad."""
    headers = {'User-Agent': UserAgent().random, 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.google.com/'}
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extracción de precio (lógica mejorada)
        price_text = "N/A"
        price_selectors = ['[class*="price"]', '[id*="price"]', '[class*="Price"]', '[id*="Price"]']
        for selector in price_selectors:
            price_tag = soup.select_one(selector)
            if price_tag:
                match = re.search(r'\d{1,3}(?:,?\d{3})*(?:\.\d{2})?', price_tag.get_text())
                if match: price_text = match.group(0).replace(',', ''); break
        
        # Extracción de imagen de alta calidad (buscando OpenGraph primero)
        image_url = ""
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            image_url = urljoin(url, og_image["content"])
        
        title = soup.title.string.strip() if soup.title else 'Sin título'
        text_content = ' '.join(soup.stripped_strings)[:1500]
        
        return {'title': title, 'text': text_content, 'price': price_text, 'image': image_url}
    except Exception as e:
        return {'title': 'N/A', 'text': '', 'price': 'N/A', 'image': ''}

def _get_relevance_score_with_gemini(query: str, product_title: str, product_text: str) -> int:
    """Usa Gemini para obtener un puntaje de relevancia numérico."""
    if not genai: return 5 # Puntaje neutral si Gemini no está disponible
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""Analyze the relevance of this product page for a user's search.
        User Search: "{query}"
        Product Page Title: "{product_title}"
        Product Page Text (excerpt): "{product_text[:500]}"
        Based on this, provide a relevance score from 1 (completely irrelevant) to 10 (perfect match).
        Respond ONLY with a single number.
        """
        response = model.generate_content(prompt)
        score = int(re.search(r'\d+', response.text).group(0))
        return min(max(score, 1), 10) # Asegurar que el score esté entre 1 y 10
    except Exception:
        return 3 # Puntaje bajo si la IA falla

def _get_suggestions_with_gemini(query: str) -> List[str]:
    """Genera búsquedas alternativas si no se encuentran resultados."""
    if not genai: return []
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"A user searched for '{query}' and found no results. Provide 3 alternative, more effective search queries for finding this product online. Respond with a JSON list of strings, like [\"query 1\", \"query 2\", \"query 3\"]."
        response = model.generate_content(prompt)
        # Limpiar y parsear la respuesta JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception:
        return []

def _get_clean_company_name(item: Dict) -> str:
    try:
        if source := item.get('source'): return source
        return urlparse(item.get('link', '')).netloc.replace('www.', '').split('.')[0].capitalize()
    except:
        return "Tienda"

@dataclass
class ProductResult:
    name: str
    price: float
    store: str
    url: str
    image_url: str = ""
    relevance_score: int = 0

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

    def get_query_from_image_vision_api(self, image_content: bytes) -> Optional[str]:
        """Búsqueda por imagen mejorada que combina múltiples etiquetas."""
        if not self.vision_client: return None
        try:
            image = vision.Image(content=image_content)
            response = self.vision_client.web_detection(image=image).web_detection
            
            all_words = []
            if response.best_guess_labels:
                all_words.extend(response.best_guess_labels[0].label.lower().split())
            if response.web_entities:
                for entity in response.web_entities[:5]: # Top 5 entidades
                    if entity.score > 0.6:
                        all_words.extend(entity.description.lower().split())
            
            if not all_words: return None
            
            stop_words = {'a', 'an', 'the', 'in', 'on', 'of', 'for', 'with', 'and', 'or', 'to'}
            word_counts = Counter(word for word in all_words if word not in stop_words and not word.isdigit() and len(word) > 2)
            # Retorna las 7 palabras más relevantes, unidas.
            return " ".join(word for word, count in word_counts.most_common(7))
        except Exception as e:
            print(f"  ❌ Fallo en análisis de imagen: {e}")
            return None

    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        """Fusiona una búsqueda de texto y una de imagen usando IA."""
        if not genai: return f"{text_query} {image_query}" # Fusión simple si no hay IA
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"A user is searching for a product. They provided this text: '{text_query}'. They also uploaded an image, from which we extracted these keywords: '{image_query}'. Combine these into a single, effective and concise search query. Respond only with the final search query."
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str]]:
        """Lógica principal de búsqueda que maneja texto, imagen o ambos."""
        text_query = query.strip() if query else None
        image_query = self.get_query_from_image_vision_api(image_content) if image_content else None
        
        final_query = None
        if text_query and image_query:
            print(f"🧠 Combinando texto '{text_query}' e imagen '{image_query}'...")
            final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query:
            final_query = text_query
        elif image_query:
            final_query = image_query

        if not final_query: print("❌ No se pudo determinar una consulta válida."); return [], []
        
        print(f"🔍 Lanzando búsqueda neuronal para: '{final_query}'")
        best_deals = self.search_with_ai_verification(final_query)
        
        suggestions = []
        if not best_deals:
            print("🤔 No se encontraron resultados. Generando sugerencias...")
            suggestions = _get_suggestions_with_gemini(final_query)

        return best_deals, suggestions

    def search_with_ai_verification(self, search_query: str) -> List[ProductResult]:
        params = {"q": search_query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "30", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
            response.raise_for_status()
            organic_results = response.json().get('organic_results', [])
            
            results_with_scores = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_item = {executor.submit(_deep_scrape_content, item.get('link')): item for item in organic_results if item.get('link')}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    content = future.result()
                    if content and content['price'] != "N/A":
                        try:
                            price_float = float(content['price'])
                            relevance_score = _get_relevance_score_with_gemini(search_query, content['title'], content['text'])
                            if relevance_score >= 5: # Umbral de relevancia mínimo
                                results_with_scores.append({
                                    'store': _get_clean_company_name(item),
                                    'product_name': item.get('title', 'Sin título'),
                                    'price_float': price_float,
                                    'url': item.get('link'),
                                    'image_url': content['image'] or item.get('thumbnail', ''),
                                    'relevance_score': relevance_score
                                })
                        except (ValueError, TypeError): continue
            
            if not results_with_scores: return []

            # Ordenamiento avanzado: primero por relevancia (mayor a menor), luego por precio (menor a mayor)
            results_with_scores.sort(key=lambda x: (-x['relevance_score'], x['price_float']))
            
            final_results_obj = [ProductResult(name=res['product_name'], price=res['price_float'], store=res['store'], url=res['url'], image_url=res.get('image_url', ''), relevance_score=res['relevance_score']) for res in results_with_scores]
            
            return final_results_obj[:30]
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda avanzada: {e}")
            return []

# ==============================================================================
# SECCIÓN 3: RUTAS FLASK Y EJECUCIÓN
# ==============================================================================
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

# ... (Rutas /login, /logout, /app sin cambios) ...

@app.route('/api/search', methods=['POST'])
def api_search():
    """Endpoint de API mejorado que devuelve resultados y sugerencias."""
    if 'user_id' not in session: return jsonify({'error': 'No autorizado'}), 401
    
    query = request.form.get('query')
    image_file = request.files.get('image_file')
    image_content = image_file.read() if image_file and image_file.filename != '' else None
    
    results, suggestions = shopping_bot.search_product(query=query, image_content=image_content)
    
    results_dicts = [res.__dict__ for res in results]
    return jsonify(results=results_dicts, suggestions=suggestions)

# ... (El resto de las rutas, plantillas y la ejecución se quedan igual) ...

# PLANTILLAS HTML
AUTH_TEMPLATE_LOGIN_ONLY = """(Pega aquí tu plantilla de Login)"""
SEARCH_TEMPLATE = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee;display:flex;flex-direction:column}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center;overflow:hidden}.product-image img{width:100%;height:100%;object-fit:cover}.product-info{padding:25px;display:flex;flex-direction:column;flex-grow:1;justify-content:space-between}.product-title{font-size:1.1em;font-weight:600;margin-bottom:12px;color:var(--text-color-dark)}.price-store-wrapper{display:flex;justify-content:space-between;align-items:center;margin-top:auto}.current-price{font-size:1.8em;font-weight:700;color:var(--accent-color)}.store-link a{font-weight:600;color:var(--primary-color);text-decoration:none}.relevance-score{position:absolute;top:10px;right:10px;background-color:rgba(0,0,0,0.6);color:white;padding:5px 10px;border-radius:12px;font-size:0.9em;font-weight:600}#suggestions{margin-top:20px;text-align:center}#suggestions h3{margin-bottom:10px}#suggestions button{background-color:#e0e0e0;border:none;padding:8px 15px;margin:5px;border-radius:8px;cursor:pointer}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas por texto?</label><input type="text" id="query" name="query" placeholder="Ej: iPhone 15 Pro, red"></div><div class="input-group"><label for="image_file">... o mejora tu búsqueda subiendo una imagen</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Buscando las mejores ofertas...</p></div><section id="results-section" class="results-section"><h2 id="results-title">Mejores Ofertas Encontradas</h2><div id="suggestions"></div><div id="products-grid" class="products-grid"></div></section></div>
<script>
const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const imageInput = document.getElementById("image_file");
const loadingDiv = document.getElementById("loading");
const resultsSection = document.getElementById("results-section");
const productsGrid = document.getElementById("products-grid");
const suggestionsDiv = document.getElementById("suggestions");

function performSearch() {
    const formData = new FormData(searchForm);
    loadingDiv.style.display = "block";
    resultsSection.style.display = "none";
    productsGrid.innerHTML = "";
    suggestionsDiv.innerHTML = "";

    fetch("{{ url_for('api_search') }}", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            loadingDiv.style.display = "none";
            if (data.results && data.results.length > 0) {
                data.results.forEach(product => {
                    productsGrid.innerHTML += `
                        <div class="product-card">
                            <div class="relevance-score" title="Puntaje de Relevancia">${product.relevance_score}/10</div>
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
            } else if (data.suggestions && data.suggestions.length > 0) {
                let suggestionsHTML = '<h3>No encontramos resultados. ¿Quizás quisiste decir...?</h3>';
                data.suggestions.forEach(suggestion => {
                    suggestionsHTML += `<button class="suggestion-btn">${suggestion}</button>`;
                });
                suggestionsDiv.innerHTML = suggestionsHTML;
                document.querySelectorAll('.suggestion-btn').forEach(button => {
                    button.addEventListener('click', () => {
                        queryInput.value = button.textContent;
                        imageInput.value = ""; // Limpiar imagen al usar sugerencia
                        document.getElementById("image-preview-container").style.display = "none";
                        performSearch();
                    });
                });
            } else {
                productsGrid.innerHTML = "<p>No se encontraron resultados para tu búsqueda.</p>";
            }
            resultsSection.style.display = "block";
        })
        .catch(error => {
            console.error("Error:", error);
            loadingDiv.style.display = "none";
            productsGrid.innerHTML = "<p>Ocurrió un error durante la búsqueda. Por favor, intenta de nuevo.</p>";
            resultsSection.style.display = "block";
        });
}

searchForm.addEventListener("submit", function(e) {
    e.preventDefault();
    performSearch();
});

imageInput.addEventListener("change", function() {
    if (this.files && this.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById("image-preview").src = e.target.result;
            document.getElementById("image-preview-container").style.display = "flex";
        };
        reader.readAsDataURL(this.files[0]);
    }
});

document.getElementById("remove-image-btn").addEventListener("click", function() {
    imageInput.value = "";
    document.getElementById("image-preview").src = "#";
    document.getElementById("image-preview-container").style.display = "none";
});
</script>
</body></html>
"""

# ... (Las rutas de autenticación y el __main__ se quedan igual) ...
