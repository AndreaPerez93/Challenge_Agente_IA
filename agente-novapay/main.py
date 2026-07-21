"""
Punto de entrada para probar el agente por consola.

Uso:
    python main.py
"""
from app.ingest import cargar_documentos, dividir_en_chunks
from app.rag import construir_vectorstore
from app.graph import construir_grafo


def main():
    print("Cargando documentos...")
    docs = cargar_documentos()
    chunks = dividir_en_chunks(docs)
    print(f"Documentos divididos en {len(chunks)} chunks.")

    print("Construyendo índice vectorial...")
    retriever = construir_vectorstore(chunks)

    print("Construyendo el grafo del agente...")
    grafo = construir_grafo(retriever)

    preguntas_de_prueba = [
        "¿Qué pasa si no acepto los términos?",
        "Tengo un problema con mi cuenta",
        "No reconozco un cargo de 5000 pesos en mi tarjeta",
    ]

    for pregunta in preguntas_de_prueba:
        resultado = grafo.invoke({"pregunta": pregunta})
        print("")
        print(f"PREGUNTA: {pregunta}")
        print(
            f"DECISION DE TRIAJE: {resultado['triaje']['decision']} | "
            f"URGENCIA: {resultado['triaje']['urgencia']} | "
            f"ACCION FINAL: {resultado['accion_final']}"
        )
        print(f"RESPUESTA: {resultado['respuesta']}")
        if resultado.get("citaciones"):
            for i, citacion in enumerate(resultado["citaciones"]):
                print(f"    - CITACION {i + 1}: {citacion.metadata.get('source')}")
        print("-" * 60)


if __name__ == "__main__":
    main()