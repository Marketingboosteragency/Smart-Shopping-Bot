# app.py (versión 12.2 - Depuración Mejorada)

# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 12.2 (Enhanced Debug Logging)
# Novedades:
# - Se añade logging detallado en cada paso del proceso de búsqueda para
#   diagnosticar el punto exacto de fallo.
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

# Configuración de APIs
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
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (ADAPTATIVA Y CON LOGGING)
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
        # GÉNESIS: Log de scraping exitoso
        print(f"    ✅ Scraping OK: {url[:50]}... (Precio encontrado: {price_text})")
        return {'title': title, 'text': text_content, 'price': price_text, 'image': image_url}
    except Exception as e:
        # GÉNESIS: Log de scraping fallido
        print(f"    ❌ Scraping Falló: {url[:50]}... Razón: {type(e).__name__}")
        return {'title': 'N/A', 'text': '', 'price': 'N/A', 'image': ''}

def _get_product_category(query: str) -> str:
    # ... (sin cambios aquí) ...

def _verify_is_product_page(query: str, page_title: str, page_content: str, category: str) -> bool:
    # ... (sin cambios aquí) ...

@dataclass
class ProductResult:
    # ... (sin cambios aquí) ...

class SmartShoppingBot:
    # ... (sin cambios aquí) ...

    def search_with_ai_verification(self, query: str, category: str) -> List[ProductResult]:
        # ... (lógica de búsqueda adaptativa) ...
        try:
            # ... (código de llamada a SerpApi) ...
            
            if blacklist:
                filtered_results = [item for item in organic_results if not any(site in item.get('link', '') for site in blacklist)]
                print(f"  📊 Resultados orgánicos: {len(organic_results)} -> Después de blacklist: {len(filtered_results)}")
            else:
                filtered_results = organic_results
                print(f"  📊 Resultados orgánicos: {len(organic_results)}")

            valid_results_after_scrape = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # ... (código del executor) ...
                    # GÉNESIS: Log de cada intento de scraping
                    print(f"  🔄 Procesando URL: {item.get('link')[:60]}...")
                    content = future.result()
                    if content and content['price'] != "N/A":
                        valid_results_after_scrape.append({'item': item, 'content': content})

            print(f"  📈 Resultados después del scraping (con precio): {len(valid_results_after_scrape)}")

            verified_results = []
            for res in valid_results_after_scrape:
                if _verify_is_product_page(query, res['content']['title'], res['content']['text'], category):
                    try:
                        price_float = float(res['content']['price'])
                        verified_results.append({'store': _get_clean_company_name(res['item']), 'product_name': res['item'].get('title', 'Sin título'), 'price_float': price_float, 'url': res['item'].get('link'), 'image_url': res['content']['image'] or res['item'].get('thumbnail', '')})
                    except (ValueError, TypeError): continue
            
            print(f"  ✅ Resultados después de la verificación de IA: {len(verified_results)}")
            
            # ... (lógica de filtrado de precios y ordenamiento) ...
            
            return final_results_obj
        except Exception as e:
            print(f"❌ Ocurrió un error en la búsqueda avanzada: {e}"); return []
