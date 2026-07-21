"""
Carga de documentos y división en chunks.

"""
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DOCUMENTS_PATH


def cargar_documentos(carpeta: str = DOCUMENTS_PATH) -> list:
    docs = []
    carpeta_path = Path(carpeta)

    if not carpeta_path.exists():
        raise FileNotFoundError(
            f"La carpeta '{carpeta}' no existe. Créala y coloca ahí tus documentos PDF."
        )

    for documento in carpeta_path.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(documento))
            docs.extend(loader.load())
            print(f"Archivo cargado: {documento.name}")
        except Exception as e:
            print(f"Error cargando archivo: {documento.name}: {e}")

    print(f"Total de documentos cargados: {len(docs)}")
    return docs


def dividir_en_chunks(docs: list, chunk_size: int = 300, chunk_overlap: int = 30) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

