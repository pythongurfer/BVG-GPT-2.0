print("--- Iniciando prueba de conexión con Ollama ---")

try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings

    # --- Prueba 1: Generar un embedding ---
    print("\n[Paso 1/2] Probando el modelo de embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    test_vector = embeddings.embed_query("Esta es una prueba.")

    # Comprueba si el vector es válido
    if test_vector and isinstance(test_vector, list) and len(test_vector) > 0:
        print("✅ ÉXITO: El embedding se generó correctamente.")
        print(f"   (Dimensiones del vector: {len(test_vector)})")
    else:
        print("❌ FALLO: No se recibió un embedding válido.")

    # --- Prueba 2: Generar una respuesta de texto ---
    print("\n[Paso 2/2] Probando el modelo de lenguaje (llama3:8b)...")
    llm = Ollama(model="llama3:8b")
    response = llm.invoke("Di 'hola' en español.")

    if response and isinstance(response, str) and len(response) > 0:
        print("✅ ÉXITO: El LLM generó una respuesta.")
        print(f"   Respuesta del modelo: '{response.strip()}'")
    else:
        print("❌ FALLO: No se recibió una respuesta válida del LLM.")

except Exception as e:
    print("\n🚨 ¡ERROR CRÍTICO DURANTE LA PRUEBA!")
    print(f"   El error es: {e}")

print("\n--- Fin de la prueba ---")