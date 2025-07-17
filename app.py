# ==============================================================================
# SECCIÓN 2: LÓGICA DEL SMART SHOPPING BOT (CON REGISTRO FORENSE)
# ==============================================================================

@dataclass
class ProductResult:
    name: str
    price: float
    store: str
    url: str
    image_url: str = ""

class SmartShoppingBot:
    def __init__(self, serpapi_key: str):
        if not serpapi_key:
            print("❌ ALERTA CRÍTICA: La variable de entorno SERPAPI_KEY no está configurada.")
        self.serpapi_key = serpapi_key

    def get_descriptive_query_from_image(self, image_content: bytes) -> Optional[str]:
        # (Esta función no necesita cambios)
        if not genai:
            print("  ❌ Análisis con Gemini Vision saltado: Modelo no configurado.")
            return None
        print("  🧠 Analizando imagen con Gemini Vision...")
        try:
            image_pil = Image.open(io.BytesIO(image_content))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = """You are an expert in identifying products. Analyze the image and generate a specific, effective search query in English to find this product for sale online. Respond ONLY with the search query."""
            response = model.generate_content([prompt, image_pil])
            query = response.text.strip().replace("*", "")
            print(f"  ✅ Consulta experta generada por Gemini Vision: '{query}'")
            return query
        except Exception as e:
            print(f"  ❌ Fallo CRÍTICO en análisis con Gemini Vision: {e}")
            return None

    def _combine_text_and_image_query(self, text_query: str, image_query: str) -> str:
        return f"{text_query} {image_query}"

    def search_product(self, query: str = None, image_content: bytes = None) -> List[ProductResult]:
        # (Esta es la función modificada con el nuevo registro)
        if not self.serpapi_key:
            print("❌ Búsqueda cancelada: La clave de API de SerpApi no está disponible.")
            return []
            
        text_query = query.strip() if query else None
        image_query = self.get_descriptive_query_from_image(image_content) if image_content else None
        
        final_query = None
        if text_query and image_query: final_query = self._combine_text_and_image_query(text_query, image_query)
        elif text_query: final_query = text_query
        elif image_query: final_query = image_query

        if not final_query:
            print("❌ No se pudo determinar una consulta de búsqueda válida.")
            return []

        print(f"🚀 Lanzando búsqueda en Google Shopping para: '{final_query}'")
        
        params = {"q": final_query, "engine": "google_shopping", "location": "United States", "gl": "us", "hl": "en", "num": "100", "api_key": self.serpapi_key}
        
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            shopping_results = data.get('shopping_results', [])

            if not shopping_results:
                print("⚠️ No se encontró la clave 'shopping_results' o estaba vacía. Respuesta de API:")
                print(json.dumps(data, indent=2))
                return []
            
            print(f"  🔍 Recibidos {len(shopping_results)} resultados brutos. Iniciando análisis forense...")
            
            # --- INICIO DEL REGISTRO FORENSE ---
            # Imprime el primer resultado completo para inspección manual
            if shopping_results:
                print("\n--- EJEMPLO DE RESULTADO BRUTO (Ítem #0) ---")
                print(json.dumps(shopping_results[0], indent=2))
                print("--------------------------------------------\n")
            # --- FIN DEL REGISTRO FORENSE ---

            products = []
            for i, item in enumerate(shopping_results):
                required_keys = ['price', 'title', 'link', 'source']
                if not all(k in item for k in required_keys):
                    missing_keys = [k for k in required_keys if k not in item]
                    # Imprime solo para los primeros 5 ítems para no saturar el log
                    if i < 5: print(f"  [Ítem #{i}] DESCARTADO: Faltan las claves: {missing_keys}")
                    continue

                try:
                    price_str = item.get('extracted_price', item['price'])
                    price_float = float(re.sub(r'[^\d.]', '', str(price_str)))
                    
                    if price_float >= 0.01:
                         products.append(ProductResult(
                            name=item['title'], price=price_float, store=item['source'],
                            url=item['link'], image_url=item.get('thumbnail', '')
                        ))
                    else:
                        if i < 5: print(f"  [Ítem #{i}] DESCARTADO: El precio ({price_float}) es demasiado bajo.")
                
                except (ValueError, TypeError) as e:
                    if i < 5: print(f"  [Ítem #{i}] DESCARTADO: Error al convertir el precio. Valor original: '{item.get('price', 'N/A')}'. Error: {e}")
                    continue
            
            print(f"✅ Análisis forense finalizado. Se procesaron {len(products)} resultados válidos.")
            products.sort(key=lambda x: x.price)
            return products

        except Exception as e:
            print(f"❌ Ocurrió un error inesperado en la búsqueda de Google Shopping: {e}")
            return []
