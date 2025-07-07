# ==============================================================================
# SMART SHOPPING BOT - APLICACIÓN COMPLETA CON FIREBASE
# Versión: 3.0
# Novedades:
# - Integración con Firebase Authentication para gestión de usuarios.
# - Registro de usuarios directamente en Firebase.
# - Inicio de sesión validado contra Firebase usando su API REST.
# - Se elimina la base de datos en memoria (`users = {}`).
# ==============================================================================

# --- IMPORTS DE LIBRERÍAS ---
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from fake_useragent import UserAgent
import os
import time
from collections import Counter

# --- IMPORTS DE FLASK Y SEGURIDAD ---
from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash

# Werkzeug ya no es necesario para hashear, Firebase lo hace por nosotros
# from werkzeug.security import generate_password_hash, check_password_hash

# --- IMPORTACIÓN DE GOOGLE CLOUD VISION ---
try:
    from google.cloud import vision

    print("✅ Módulo de Google Cloud Vision importado.")
except ImportError:
    print("❌ ERROR: El módulo 'google-cloud-vision' no está instalado. Ejecuta: pip install google-cloud-vision")
    vision = None

# --- NUEVO: IMPORTACIÓN DE FIREBASE ADMIN SDK ---
try:
    import firebase_admin
    from firebase_admin import credentials, auth

    print("✅ Módulo de Firebase Admin importado.")
except ImportError:
    print("❌ ERROR: El módulo 'firebase-admin' no está instalado. Ejecuta: pip install firebase-admin")
    firebase_admin = None

# --- CÓDIGO CON LA SOLUCIÓN PARA ANDROID/PYDROID 3 ---
# En lugar de adivinar la ruta, le damos la ruta fija y absoluta a la carpeta del proyecto.

project_dir = "/storage/emulated/0/Mi_Shopping_Bot" 

# Google Vision
google_credentials_path = os.path.join(project_dir, 'credentials', 'google-credentials.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

# Firebase
firebase_credentials_path = os.path.join(project_dir, 'credentials', 'firebase-credentials.json')
# --- FIN DEL BLOQUE DE SOLUCIÓN ---

# =============== CONFIGURACIÓN DE APIs ===============
SERPAPI_KEY = "17173965a2e2531e0a06b3864a1138d65200f7b238188cd3a5e0fe9b2f1b36a9"  # ← TU API KEY DE SERPAPI

# --- NUEVO: CONFIGURACIÓN DE FIREBASE ---
# PEGA AQUÍ LA WEB API KEY QUE COPIASTE DEL PASO 2B
FIREBASE_WEB_API_KEY = "AIzaSyBXh3nJ5uxAkZbRgQZtbGUEODd-A6vYjxk"

# Bloque MODIFICADO para depurar
if firebase_admin:
    try:
        print("▶️ Intentando inicializar Firebase Admin...")
        if not os.path.exists(firebase_credentials_path):
             raise FileNotFoundError(f"El archivo de credenciales de Firebase no se encontró en: {firebase_credentials_path}")
        cred = credentials.Certificate(firebase_credentials_path)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK inicializado correctamente.")
    except Exception as e:
        # Añadimos un print de error más visible
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!!!!!!!!!  ERROR CRÍTICO EN FIREBASE INIT  !!!!!!!!!!")
        print(f"❌ ERROR: No se pudo inicializar Firebase Admin SDK. Detalles: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        firebase_admin = None  # Desactivar si falla la inicialización


# ==============================================================================
# SECCIÓN 1: LÓGICA DEL SMART SHOPPING BOT (VERSIÓN FINAL Y COMPLETA)
# ==============================================================================
@dataclass
class ProductResult:
    name: str
    price: float
    store: str
    url: str
    image_url: str = ""
    rating: float = 0.0
    reviews: int = 0
    availability: str = "In Stock"
    shipping: str = ""
    original_price: float = 0.0
    discount: str = ""
    seller: str = ""


class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.base_url = "https://serpapi.com/search"
        self.ua = UserAgent()
        self.STOP_WORDS = set(
            ['a', 'an', 'the', 'and', 'or', 'in', 'on', 'for', 'with', 'by', 'of', 'to', 'new', 'sale', 'buy', 'com',
             'www', 'https', 'http', 'price', 'free', 'shipping', 'review', 'image', 'stock', 'photo', 'picture'])

        # Bloque de inicialización de Google Vision
        if vision and os.path.exists(google_credentials_path):
            try:
                print("▶️ Intentando inicializar Google Cloud Vision...")
                self.vision_client = vision.ImageAnnotatorClient()
                print("✅ Cliente de Google Cloud Vision inicializado correctamente.")
            except Exception as e:
                print(f"❌ ERROR CRÍTICO EN VISION INIT: {e}")
                self.vision_client = None
        else:
            self.vision_client = None

    def _get_intelligent_query_from_vision(self, annotations) -> Optional[str]:
        if not annotations: return None
        all_words = []
        if annotations.best_guess_labels:
            all_words.extend(re.sub(r'[^\w\s]', '', annotations.best_guess_labels[0].label.lower()).split())
        if annotations.web_entities:
            for entity in annotations.web_entities[:3]:
                if entity.score > 0.6:
                    all_words.extend(re.sub(r'[^\w\s]', '', entity.description.lower()).split())
        if not all_words: return None
        word_counts = Counter(word for word in all_words if word not in self.STOP_WORDS and not word.isdigit() and len(word) > 2)
        return " ".join(word for word, count in word_counts.most_common(6)) if word_counts else None

    def get_query_from_image_vision_api(self, image_path: str) -> Optional[str]:
        if not self.vision_client: return None
        print("  🧠 Analizando imagen con Google Cloud Vision API...")
        try:
            with open(image_path, "rb") as image_file: content = image_file.read()
            image = vision.Image(content=content)
            response = self.vision_client.web_detection(image=image)
            return self._get_intelligent_query_from_vision(response.web_detection)
        except Exception as e:
            print(f"  ❌ Fallo en análisis con Google Cloud Vision: {e}")
            return None

    def search_product(self, query: str = None, image_path: str = None) -> Tuple[List[ProductResult], bool]:
        print("\n🚀 Iniciando Smart Shopping Bot...")
        final_query = query
        if image_path:
            final_query = self.get_query_from_image_vision_api(image_path)
        if not final_query:
            print("❌ No se pudo determinar una consulta válida.")
            return [], False
        
        print(f"🔍 Lanzando búsqueda para: '{final_query}'")
        all_results = self.comprehensive_search(final_query)
        best_deals = self.get_best_deals(all_results)
        return best_deals, False

    def _make_request(self, params: Dict, timeout: int = 30) -> Dict:
        try:
            params['api_key'] = self.serpapi_key
            response = requests.get(self.base_url, params=params, headers={'User-Agent': self.ua.random}, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            print(f"\n--- RESPUESTA JSON DE {params.get('engine')} RECIBIDA ---\n" + str(data)[:500] + "...\n----------------------\n")
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error en petición API ({params.get('engine')}): {e}")
            return {}

    def comprehensive_search(self, query: str, max_per_store: int = 5) -> Dict[str, List[ProductResult]]:
        search_functions = [
            ('Google Shopping', self.search_google_shopping),
            ('Amazon', self.search_amazon),
            ('Walmart', self.search_walmart),
            ('eBay', self.search_ebay),
            ('Home Depot', self.search_home_depot)
        ]
        all_results = {}
        with ThreadPoolExecutor(max_workers=len(search_functions)) as executor:
            future_to_store = {executor.submit(func, query, max_per_store): name for name, func in search_functions}
            for future in as_completed(future_to_store):
                store_name = future_to_store[future]
                try:
                    results = future.result()
                    all_results[store_name] = results
                    print(f"✓ {store_name}: {len(results)} productos encontrados.")
                except Exception as e:
                    print(f"✗ Error en {store_name}: {e}")
                    all_results[store_name] = []
        return all_results

    def get_best_deals(self, all_results: Dict[str, List[ProductResult]], top_n: int = 30) -> List[ProductResult]:
        all_products = [p for store_results in all_results.values() for p in store_results]
        valid_products = [p for p in all_products if p.price > 0]
        valid_products.sort(key=lambda x: x.price)
        print(f"✅ Se encontraron {len(valid_products)} ofertas válidas en total.")
        return valid_products[:top_n]

    def search_google_shopping(self, query: str, max_results: int = 5) -> List[ProductResult]:
        params = {'engine': 'google_shopping', 'q': query, 'num': max_results}
        data = self._make_request(params)
        results = []
        for item in data.get('shopping_results', []):
            try:
                price_str = item.get('extracted_price') or (item.get('price', {}).get('value')) or item.get('price')
                if price_str:
                    price = float(re.sub(r'[^\d.]', '', str(price_str)))
                    if price > 0:
                        results.append(ProductResult(name=item.get('title', 'N/A'), price=price, store=item.get('source', 'Google'), url=item.get('link', ''), image_url=item.get('thumbnail', '')))
                        print(f"  -> Producto Google: {item.get('title', 'N/A')[:30]}... ${price}")
            except (ValueError, KeyError, TypeError):
                continue
        return results

    def search_amazon(self, query: str, max_results: int = 5) -> List[ProductResult]:
        params = {'engine': 'amazon', 'k': query, 'amazon_domain': 'amazon.com', 'num': max_results}
        data = self._make_request(params)
        results = []
        for item in data.get('organic_results', []):
            try:
                price_info = item.get('price')
                if price_info and price_info.get('raw'):
                    price = float(re.sub(r'[^\d.]', '', price_info['raw']))
                    if price > 0 and item.get('asin'):
                        results.append(ProductResult(name=item.get('title', 'N/A'), price=price, store='Amazon', url=f"https://www.amazon.com/dp/{item.get('asin')}", image_url=item.get('image', '')))
                        print(f"  -> Producto Amazon: {item.get('title', 'N/A')[:30]}... ${price}")
            except (ValueError, KeyError, TypeError):
                continue
        return results

    def search_walmart(self, query: str, max_results: int = 5) -> List[ProductResult]: return []
    def search_ebay(self, query: str, max_results: int = 5) -> List[ProductResult]: return []
    def search_home_depot(self, query: str, max_results: int = 5) -> List[ProductResult]: return []
# ==============================================================================
# SECCIÓN 2: CONFIGURACIÓN DE FLASK
# ==============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'una-clave-secreta-muy-fuerte-y-dificil-de-adivinar'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
shopping_bot = SmartShoppingBot(SERPAPI_KEY)

# --- ELIMINADO: La base de datos en memoria ya no es necesaria ---
# users = {}


# ==============================================================================
# SECCIÓN 3: PLANTILLAS HTML (SIN CAMBIOS SIGNIFICATIVOS)
# [Pega aquí las mismas plantillas AUTH_TEMPLATE y SEARCH_TEMPLATE del código anterior]
# ...
AUTH_TEMPLATE = """ <!DOCTYPE html>... """  # Tu HTML de autenticación aquí
SEARCH_TEMPLATE = """ <!DOCTYPE html>... """  # Tu HTML de búsqueda aquí


# ==============================================================================


# ==============================================================================
# SECCIÓN 4: RUTAS DE LA APLICACIÓN FLASK (MODIFICADAS PARA FIREBASE)
# ==============================================================================

# --- Rutas de Autenticación ---

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('main_app_page'))
    return render_template_string(AUTH_TEMPLATE)


@app.route('/login', methods=['POST'])
def login():
    """ MODIFICADO: Procesa el inicio de sesión contra la API REST de Firebase. """
    email = request.form.get('email')
    password = request.form.get('password')

    # URL de la API REST de Firebase para iniciar sesión
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"

    # Datos a enviar a Firebase
    payload = {
        'email': email,
        'password': password,
        'returnSecureToken': True
    }

    try:
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status()  # Lanza un error si la respuesta no es 2xx

        user_data = response.json()

        # Guardamos datos importantes en la sesión de Flask
        session['user_id'] = user_data['localId']  # ID único de Firebase
        session['user_name'] = user_data.get('displayName', email)  # Nombre del usuario
        session['id_token'] = user_data['idToken']  # Token para futuras llamadas seguras

        flash('¡Has iniciado sesión correctamente!', 'success')
        return redirect(url_for('main_app_page'))

    except requests.exceptions.HTTPError as e:
        # Firebase devuelve errores detallados en el JSON de la respuesta
        error_json = e.response.json().get('error', {})
        error_message = error_json.get('message', 'ERROR_DESCONOCIDO')

        if error_message == 'INVALID_PASSWORD' or error_message == 'EMAIL_NOT_FOUND':
            flash('Correo o contraseña incorrectos.', 'danger')
        else:
            flash(f'Error al iniciar sesión: {error_message}', 'danger')

        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Ocurrió un error inesperado: {e}', 'danger')
        return redirect(url_for('index'))


@app.route('/register', methods=['POST'])
def register():
    """ MODIFICADO: Registra un nuevo usuario usando el SDK de Firebase Admin. """
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    if not firebase_admin:
        flash('El servicio de autenticación no está disponible. Contacta al administrador.', 'danger')
        return redirect(url_for('index'))

    try:
        # Crea el usuario en Firebase Authentication
        user = auth.create_user(
            email=email,
            password=password,
            display_name=name,
            email_verified=False  # Puedes implementar un flujo de verificación de correo
        )
        print(f"Usuario creado en Firebase con UID: {user.uid}")
        flash('¡Registro exitoso! Ahora puedes iniciar sesión.', 'success')
    except auth.EmailAlreadyExistsError:
        flash('El correo electrónico ya está registrado.', 'warning')
    except ValueError as e:
        # Captura errores comunes como contraseña débil
        flash(f'Error en el registro: {e}', 'danger')
    except Exception as e:
        flash(f'Ocurrió un error inesperado durante el registro: {e}', 'danger')

    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.clear()  # Limpia toda la sesión
    flash('Has cerrado la sesión.', 'success')
    return redirect(url_for('index'))


# --- Rutas de la Aplicación Principal (Protegidas) ---
@app.route('/app')
def main_app_page():
    if 'user_id' not in session:
        flash('Debes iniciar sesión para acceder a esta página.', 'warning')
        return redirect(url_for('index'))

    # MODIFICADO: Obtenemos el nombre de la sesión, no de la base de datos local
    user_name = session.get('user_name', 'Usuario')
    return render_template_string(SEARCH_TEMPLATE, user_name=user_name)


# (La ruta de API search no necesita cambios, ya que su seguridad depende de la sesión de Flask)
@app.route('/api/search', methods=['POST'])
def api_search():
    if 'user_id' not in session:
        return jsonify({'error': 'No autorizado'}), 401

    query = request.form.get('query')
    image_file = request.files.get('image_file')
    image_path = None

    if image_file and image_file.filename != '':
        filename = f"{os.urandom(8).hex()}_{image_file.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

    results, is_alternative = shopping_bot.search_product(query=query, image_path=image_path)

    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    results_dicts = [p.__dict__ for p in results]
    return jsonify(results=results_dicts, is_alternative=is_alternative)


# ==============================================================================
# SECCIÓN 5: EJECUCIÓN DE LA APLICACIÓN
# ==============================================================================

    AUTH_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Acceso | Smart Shopping Bot</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--text-color-dark:#2C3E50;--card-bg:#FFFFFF;--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:linear-gradient(135deg,var(--primary-color) 0%,var(--secondary-color) 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px}.auth-container{max-width:480px;width:100%;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-medium);overflow:hidden;animation:fadeIn .8s ease-out}@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}.form-header{text-align:center;padding:40px 30px 20px}.form-header h1{color:var(--text-color-dark);font-size:2em;margin-bottom:10px}.form-header p{color:#7f8c8d;font-size:1.1em}.form-toggle{display:flex;background-color:#f0f4f8;border-radius:12px;margin:25px auto;padding:6px;max-width:300px}.toggle-btn{flex:1;padding:12px;border:none;background-color:transparent;color:#8a99ab;font-size:1em;font-weight:600;border-radius:8px;cursor:pointer;transition:all .3s ease}.toggle-btn.active{background-color:var(--card-bg);color:var(--primary-color);box-shadow:0 4px 10px rgba(0,0,0,.1)}.form-body{padding:10px 40px 40px}form{display:flex;flex-direction:column;gap:20px}form.hidden{display:none}.input-group{display:flex;flex-direction:column;gap:8px}.input-group label{font-weight:600;color:var(--text-color-dark);font-size:.95em}.input-group input{padding:16px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:16px;transition:all .3s ease}.input-group input:focus{outline:0;border-color:var(--primary-color);box-shadow:0 0 0 4px rgba(74,144,226,.2)}.submit-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:16px 30px;font-size:1.1em;font-weight:600;border-radius:12px;cursor:pointer;transition:all .3s ease;margin-top:15px}.submit-btn:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(0,0,0,.2)}.flash-messages{list-style:none;padding:0 40px 20px}.flash{padding:15px;margin-bottom:15px;border-radius:8px;text-align:center}.flash.success{background-color:#d4edda;color:#155724}.flash.danger{background-color:#f8d7da;color:#721c24}.flash.warning{background-color:#fff3cd;color:#856404}</style></head><body><div class="auth-container"><div class="form-header"><h1 id="form-title">Bienvenido de Nuevo</h1><p id="form-subtitle">Accede para encontrar las mejores ofertas.</p></div>{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}<ul class=flash-messages>{% for category, message in messages %}<li class="flash {{ category }}">{{ message }}</li>{% endfor %}</ul>{% endif %}{% endwith %}<div class="form-toggle"><button id="show-login-btn" class="toggle-btn active">Iniciar Sesión</button><button id="show-register-btn" class="toggle-btn">Registrarse</button></div><div class="form-body"><form id="login-form" action="{{ url_for('login') }}" method="post"><div class="input-group"><label for="login-email">Correo</label><input type="email" name="email" required></div><div class="input-group"><label for="login-password">Contraseña</label><input type="password" name="password" required></div><button type="submit" class="submit-btn">Entrar</button></form><form id="register-form" action="{{ url_for('register') }}" method="post" class="hidden"><div class="input-group"><label>Nombre</label><input type="text" name="name" required></div><div class="input-group"><label>Correo</label><input type="email" name="email" required></div><div class="input-group"><label>Contraseña</label><input type="password" name="password" required></div><button type="submit" class="submit-btn">Crear Cuenta</button></form></div></div><script>const showLoginBtn=document.getElementById("show-login-btn"),showRegisterBtn=document.getElementById("show-register-btn"),loginForm=document.getElementById("login-form"),registerForm=document.getElementById("register-form"),formTitle=document.getElementById("form-title"),formSubtitle=document.getElementById("form-subtitle");showLoginBtn.addEventListener("click",()=>{showLoginBtn.classList.add("active"),loginForm.classList.remove("hidden"),showRegisterBtn.classList.remove("active"),registerForm.classList.add("hidden"),formTitle.textContent="Bienvenido de Nuevo",formSubtitle.textContent="Accede para encontrar las mejores ofertas."}),showRegisterBtn.addEventListener("click",()=>{showRegisterBtn.classList.add("active"),registerForm.classList.remove("hidden"),showLoginBtn.classList.remove("active"),loginForm.classList.add("hidden"),formTitle.textContent="Crea tu Cuenta",formSubtitle.textContent="Únete y empieza a comprar de forma inteligente."});</script></body></html>
    """
    SEARCH_TEMPLATE = """
    <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Smart Shopping Bot - Comparador de Precios</title><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet"><style>:root{--primary-color:#4A90E2;--secondary-color:#50E3C2;--accent-color:#FF6B6B;--text-color-dark:#2C3E50;--text-color-light:#ECF0F1;--bg-light:#F8F9FA;--card-bg:#FFFFFF;--shadow-light:rgba(0,0,0,0.08);--shadow-medium:rgba(0,0,0,0.15)}body{font-family:'Poppins',sans-serif;background:var(--bg-light);min-height:100vh;padding:20px;color:var(--text-color-dark)}.container{max-width:1400px;width:100%;margin:0 auto;background:var(--card-bg);border-radius:20px;box-shadow:0 25px 50px var(--shadow-light);overflow:hidden}.header{background:linear-gradient(45deg,var(--text-color-dark),var(--primary-color));color:var(--text-color-light);padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.1em;opacity:.9}.header a{color:var(--secondary-color);text-decoration:none;font-weight:600}.search-section{padding:50px;background:var(--bg-light);border-bottom:1px solid #e0e0e0}.search-form{display:flex;flex-direction:column;gap:25px;max-width:700px;margin:0 auto}.input-group{display:flex;flex-direction:column;gap:12px}.input-group label{font-weight:600;font-size:1.1em}.input-group input{padding:18px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:17px}.search-btn{background:linear-gradient(45deg,var(--primary-color),#2980b9);color:#fff;border:none;padding:18px 35px;font-size:1.2em;font-weight:600;border-radius:12px;cursor:pointer}.loading{text-align:center;padding:60px;display:none}.spinner{border:5px solid rgba(74,144,226,.2);border-top:5px solid var(--primary-color);border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:0 auto 30px}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.results-section{padding:50px;display:none}.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px;margin-top:40px}.product-card{background:var(--card-bg);border-radius:18px;box-shadow:0 12px 30px var(--shadow-light);overflow:hidden;border:1px solid #eee}.product-image{width:100%;height:220px;display:flex;align-items:center;justify-content:center}.product-image img{max-width:90%;max-height:90%;object-fit:contain}.product-info{padding:25px}.product-title{font-size:1.25em;font-weight:600;margin-bottom:12px}.current-price{font-size:2.2em;font-weight:700;color:var(--accent-color)}#image-preview-container{display:none;align-items:center;gap:20px;margin-top:20px}#image-preview{max-height:100px;border-radius:10px}#remove-image-btn{background:var(--accent-color);color:#fff;border:none;border-radius:50%;width:35px;height:35px;cursor:pointer}</style></head><body><div class="container"><header class="header"><h1>Smart Shopping Bot</h1><p>Hola, <strong>{{ user_name }}</strong>. Encuentra los mejores precios online. | <a href="{{ url_for('logout') }}">Cerrar Sesión</a></p></header><section class="search-section"><form id="search-form" class="search-form"><div class="input-group"><label for="query">¿Qué producto buscas por texto?</label><input type="text" id="query" name="query" placeholder="Ej: iPhone 15 Pro"></div><div class="input-group"><label for="image_file">... o busca subiendo una imagen</label><input type="file" id="image_file" name="image_file" accept="image/*"><div id="image-preview-container"><img id="image-preview" src="#" alt="Previsualización"><button type="button" id="remove-image-btn" title="Eliminar imagen">×</button></div></div><button type="submit" id="search-btn" class="search-btn">Buscar Precios</button></form></section><div id="loading" class="loading"><div class="spinner"></div><p>Buscando las mejores ofertas...</p></div><section id="results-section" class="results-section"><h2 id="results-title">Mejores Ofertas Encontradas</h2><div id="products-grid" class="products-grid"></div></section></div><script>const searchForm=document.getElementById("search-form");searchForm.addEventListener("submit",function(e){e.preventDefault();const t=new FormData(searchForm);document.getElementById("loading").style.display="block",document.getElementById("results-section").style.display="none",fetch("{{ url_for('api_search') }}",{method:"POST",body:t}).then(e=>e.json()).then(e=>{document.getElementById("loading").style.display="none";const t=document.getElementById("products-grid");if(t.innerHTML="",e.results&&e.results.length>0)e.results.forEach(e=>{t.innerHTML+=`\n                            <div class="product-card">\n                                <div class="product-image"><img src="${e.image_url||""}" alt="${e.name}"></div>\n                                <div class="product-info">\n                                    <div class="product-title">${e.name}</div>\n                                    <div class="current-price">$${e.price.toFixed(2)}</div>\n                                    <div><a href="${e.url}" target="_blank">Ver en ${e.store}</a></div>\n                                </div>\n                            </div>`});else t.innerHTML="<p>No se encontraron resultados.</p>";document.getElementById("results-section").style.display="block"}).catch(e=>{console.error("Error:",e),document.getElementById("loading").style.display="none"})}),document.getElementById("image_file").addEventListener("change",function(){if(this.files&&this.files[0]){var e=new FileReader;e.onload=function(e){document.getElementById("image-preview").src=e.target.result,document.getElementById("image-preview-container").style.display="flex"},e.readAsDataURL(this.files[0])}}),document.getElementById("remove-image-btn").addEventListener("click",function(){document.getElementById("image_file").value="",document.getElementById("image-preview").src="#",document.getElementById("image-preview-container").style.display="none"});</script></body></html>
    """
    app.run(debug=True, host='0.0.0.0', port=5000)