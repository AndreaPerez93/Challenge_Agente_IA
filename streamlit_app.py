"""
Interfaz de chat web simple para el agente de NovaPay, usando Streamlit.

Uso:
    streamlit run streamlit_app.py
"""
import streamlit as st

from app.ingest import cargar_documentos, dividir_en_chunks
from app.rag import construir_vectorstore
from app.graph import construir_grafo

st.set_page_config(page_title="Agente NovaPay", page_icon="🔐")

AVATAR_USUARIO = "🧑"
AVATAR_AGENTE = "🤖"


@st.cache_resource(show_spinner="Cargando documentos y preparando el agente...")
def cargar_agente():
    docs = cargar_documentos()
    chunks = dividir_en_chunks(docs)
    retriever = construir_vectorstore(chunks)
    return construir_grafo(retriever)


grafo = cargar_agente()

st.title("😎 Agente NovaPay")
st.caption(
    "Estás conversando con un agente de inteligencia artificial, no con una persona. "
    "Responde con base en los documentos internos de NovaPay."
)

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial de la conversación
for i, mensaje in enumerate(st.session_state.historial):
    avatar = AVATAR_USUARIO if mensaje["role"] == "user" else AVATAR_AGENTE
    with st.chat_message(mensaje["role"], avatar=avatar):
        st.markdown(mensaje["content"])

        if mensaje["role"] == "assistant":
            if mensaje.get("citaciones"):
                with st.expander("📄 Fuentes utilizadas"):
                    for c in mensaje["citaciones"]:
                        fuente = c.metadata.get("source", "desconocida")
                        st.markdown(f"- `{fuente}`")

            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                st.button("👍", key=f"like_{i}")
            with col2:
                st.button("👎", key=f"dislike_{i}")

# Campo de entrada
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar=AVATAR_USUARIO):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar=AVATAR_AGENTE):
        with st.spinner("Pensando..."):
            resultado = grafo.invoke({"pregunta": pregunta})

        st.markdown(resultado["respuesta"])

        if resultado.get("citaciones"):
            with st.expander("📄 Fuentes utilizadas"):
                for c in resultado["citaciones"]:
                    fuente = c.metadata.get("source", "desconocida")
                    st.markdown(f"- `{fuente}`")

    st.session_state.historial.append(
        {
            "role": "assistant",
            "content": resultado["respuesta"],
            "citaciones": resultado.get("citaciones", []),
        }
    )
    st.rerun()