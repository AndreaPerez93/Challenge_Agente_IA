import os
from dotenv import load_dotenv

load_dotenv()  # busca un archivo .env en la raíz del proyecto

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "No se encontró GEMINI_API_KEY. Copia el archivo .env.example como .env "
        "y agrega ahí tu llave de la API de Gemini (https://aistudio.google.com/apikey)."
    )

# Carpeta donde se buscarán los documentos a indexar.
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "documents")