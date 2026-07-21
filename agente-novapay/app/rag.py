"""
Núcleo del RAG: modelo de lenguaje, embeddings, vectorstore y cadena de respuesta.
"""
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.config import GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)

modelo_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)

PROMPT_RAG = ChatPromptTemplate(
    [
        (
            "system",
            """
            Eres el especialista en políticas internas de la empresa NovaPay.
            Responde siempre utilizando los conocimientos del contexto que te fue pasado.
            Si no hay información sobre la pregunta en el contexto, responde sólo 'No tengo respuesta a esa pregunta.'.
            """,
        ),
        ("human", "Contexto: {context}.\nPregunta del colaborador: {input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, PROMPT_RAG)


def construir_vectorstore(chunks: list):
    """Crea el índice FAISS a partir de los chunks y devuelve un retriever listo para usar."""
    vectorstore = FAISS.from_documents(chunks, modelo_embeddings)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4},
    )


def busqueda_de_respuestas_rag(pregunta: str, retriever) -> dict:
    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {"respuesta": "No tengo respuesta a esa pregunta.", "citaciones": [], "documentos_encontrados": False}

    answer = document_chain.invoke({"input": pregunta, "context": documentos_relacionados})

    if answer.rstrip(".!?") == "No tengo respuesta a esa pregunta.":
        return {"respuesta": "No tengo respuesta a esa pregunta.", "citaciones": [], "documentos_encontrados": False}

    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True,
    }