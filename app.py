# app.py (versión 7.0 - Integración con OpenAI CLIP)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 7.0 (OpenAI CLIP Integration)
# Novedades:
# - Se reemplaza Google Cloud Vision por OpenAI CLIP para el análisis de imágenes.
# - El modelo CLIP se carga una vez en memoria al iniciar la aplicación.
# - La búsqueda por imagen ahora clasifica la imagen contra una lista de texto predefinida.
# - ADVERTENCIA: Esta versión requiere muchos recursos (RAM) y puede no funcionar
#   en el plan gratuito de Render.
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
from collections import Counter
from PIL import Image

# --- IMPORTS DE APIS DE IA ---
# GÉNESIS: Imports para CLIP de OpenAI
try:
    import torch
    import clip
    print("✅ Módulos de PyTorch y CLIP importados.")
except ImportError:
    print("❌ ERROR: 'torch' o 'clip' no están instalados. Revisa requirements.txt.")
    torch = None
    clip = None

try:
    import google.generativeai as genai
    print("✅ Módulo de Google Generative AI (Gemini) importado.")
except ImportError:
    print("⚠️ AVISO: 'google-generativeai' no está instalado.")
    genai = None

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL DE FLASK, APIS Y MODELOS DE IA
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

# GÉNESIS: Carga del modelo CLIP al iniciar la aplicación
clip_model, clip_preprocess = None, None
if torch and clip:
    try:
        print("🧠 Cargando modelo CLIP en memoria (esto puede tardar y consumir mucha RAM)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("✅ Modelo CLIP cargado exitosamente.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar modelo CLIP: {e}")
        # La aplicación puede continuar sin CLIP, pero la búsqueda por imagen fallará.

# GÉNESIS: Define aquí las categorías de productos para que CLIP las reconozca.
CLIP_TEXT_LABELS = [
    "pieza metálica para motor", "cárter de aceite de automóvil", "pieza de carro",
    "pieza de maquinaria", "bandeja de cocina", "recipiente de laboratorio",
    "un teléfono inteligente", "una computadora portátil", "un accesorio de moda",
    "un bolso de mano", "un zapato", "muebles para el hogar", "un vestido de mujer",
    "una camiseta de hombre", "un juguete para niños", "un libro", "equipo deportivo"
]

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (CON CLIP)
# ==============================================================================

# ... (las funciones _deep_scrape_content, _get_relevance_score_with_gemini, etc., se quedan igual) ...

@dataclass
class ProductResult:
    # ... (sin cambios)
    name: str; price: float; store: str; url: str; image_url: str = ""; relevance_score: int = 0

class SmartShoppingBot:
    def __init__(self, serpapi_key: str, clip_model_tuple: tuple):
        self.serpapi_key = serpapi_key
        # GÉNESIS: Pasamos el modelo y el preprocesador de CLIP
        self.clip_model, self.clip_preprocess = clip_model_tuple

    def get_query_from_clip_api(self, image_content: bytes) -> Optional[str]:
        """Usa CLIP para encontrar la mejor descripción de una imagen."""
        if not self.clip_model or not self.clip_preprocess:
            print("  ❌ Análisis con CLIP saltado: Modelo no cargado.")
            return None
        
        print("  🧠 Analizando imagen con OpenAI CLIP...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            image = self.clip_preprocess(Image.open(io.BytesIO(image_content))).unsqueeze(0).to(device)
            text_inputs = clip.tokenize(CLIP_TEXT_LABELS).to(device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalizar features para un cálculo de similaridad coseno más preciso
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_prob_index = similarity[0].argmax().item()
                best_label = CLIP_TEXT_LABELS[best_prob_index]
                
                print(f"  ✅ Mejor descripción encontrada por CLIP: '{best_label}'")
                return best_label
        except Exception as e:
            print(f"  ❌ Fallo en análisis con CLIP: {e}")
            return None

    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        # ... (sin cambios)
        if not genai: return f"{text_query} {image_query}"
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"A user is searching for a product. Text: '{text_query}'. Keywords from image: '{image_query}'. Combine these into a single, effective search query. Respond only with the final query."
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> Tuple[List[ProductResult], List[str]]:
        text_query = query.strip() if query else None
        # GÉNESIS: Se llama a la nueva función de CLIP
        image_query = self.get_query_from_clip_api(image_content) if image_content else None
        
        final_query = None
        if text_query and image_query:
            print(f"🧠 Combinando texto '{text_query}' e imagen (descripción CLIP: '{image_query}')...")
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
        # ... (código optimizado de la versión 6.2 se queda igual) ...
        params = {"q": search_query, "engine": "google", "location": "United States", "gl": "us", "hl": "en", "num": "20", "api_key": self.serpapi_key}
        try:
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
# GÉNESIS: Pasamos el modelo cargado al crear la instancia del bot
shopping_bot = SmartShoppingBot(SERPAPI_KEY, (clip_model, clip_preprocess))

# ... (todas las rutas de Flask y las plantillas HTML se quedan exactamente igual que en la v6.2) ...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
