# app.py (versión con mejoras críticas de IA y scraping)

# ... (todos los imports iniciales se quedan igual) ...
import requests, re, json, os, time, statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash
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

# ... (la configuración de APIs se queda igual) ...
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get('GOOGLE_CREDENTIALS_JSON')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Google Generative AI (Gemini) configurada.")
    except Exception as e:
        print(f"❌ ERROR al configurar API de Gemini: {e}")
        genai = None
# ... (la inicialización de Vision se queda igual) ...

# ==============================================================================
# SECCIÓN 1: LÓGICA DEL SMART SHOPPING BOT (MODIFICADA)
# ==============================================================================

def _deep_scrape_content(url: str) -> Dict:
    # GÉNESIS: Headers mejorados para simular un navegador real y evitar bloqueos.
    headers = {
        'User-Agent': UserAgent().random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1'
    }
    try:
        # GÉNESIS: Usar una sesión para manejar cookies, que ayuda con los bloqueos.
        s = requests.Session()
        response = s.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # GÉNESIS: Lógica de extracción de precio ligeramente mejorada (aún un placeholder).
        price_text = "N/A"
        # Busca patrones más específicos de precio
        price_patterns = [
            r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', # $1,234.56 o $1234.56
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s?USD' # 1,234.56 USD
        ]
        for pattern in price_patterns:
            price_tags = soup.find_all(text=re.compile(pattern))
            if price_tags:
                match = re.search(pattern, price_tags[0])
                if match:
                    price_text = match.group(1).replace(',', '')
                    break # Salimos al encontrar el primer precio

        title = soup.title.string if soup.title else 'Sin título'
        text_content = ' '.join(soup.stripped_strings)[:1000]
        print(f"    Scraping en {url[:40]}... Título: '{title[:30]}...', Precio: {price_text}")
        return {'title': title, 'text': text_content, 'price': price_text}
    except requests.exceptions.HTTPError as http_err:
        print(f"    Falló el scraping en {url[:40]}: Error HTTP {http_err.response.status_code}")
        return {'title': 'N/A', 'text': '', 'price': 'N/A'}
    except Exception as e:
        print(f"    Falló el scraping en {url[:40]}: {e}")
        return {'title': 'N/A', 'text': '', 'price': 'N/A'}

def _verify_product_with_gemini(query: str, product_title: str, product_text: str) -> bool:
    if not genai:
        print("    (Saltando verificación IA: Gemini no configurado)")
        return True
    try:
        # GÉNESIS: SOLUCIÓN - Usar un modelo actual y disponible.
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""Is the following product relevant to the user's search? Answer only with 'SI' or 'NO'. User search: "{query}". Page title: "{product_title}". Page text extract: "{product_text[:500]}" """
        response = model.generate_content(prompt)
        decision = response.text.strip().upper()
        print(f"    Verificación IA para '{product_title[:30]}...': {decision}")
        return "SI" in decision
    except Exception as e:
        print(f"    Error en verificación IA: {e}")
        # GÉNESIS: Si la IA falla, es mejor asumir que el producto NO es relevante
        # para no mostrar resultados basura.
        return False

# ... (resto de la clase SmartShoppingBot y las rutas de Flask se quedan exactamente igual) ...
# (El código desde _get_clean_company_name hasta el final no cambia)
