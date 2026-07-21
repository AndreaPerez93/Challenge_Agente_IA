"""
Define el estado del agente y arma el grafo con LangGraph:
triaje -> (responder_automatico | solicitar_aclaracion | escalar_a_soporte)
"""
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from app.rag import busqueda_de_respuestas_rag
from app.triaje import triaje


class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    rag_exito: bool
    accion_final: str


def construir_grafo(retriever):
    def nodo_triaje(state: AgentState) -> AgentState:
        print("Ejecutando nodo 'triaje'...")
        return {"triaje": triaje(state["pregunta"])}

    def nodo_responder_automatico(state: AgentState) -> AgentState:
        print("Ejecutando nodo 'responder_automatico'...")
        respuesta_rag = busqueda_de_respuestas_rag(state["pregunta"], retriever)

        update: AgentState = {
            "respuesta": respuesta_rag["respuesta"],
            "citaciones": respuesta_rag["citaciones"],
            "rag_exito": respuesta_rag["documentos_encontrados"],
        }

        if respuesta_rag["documentos_encontrados"]:
            update["accion_final"] = "RESPONDER_AUTOMATICO"

        return update

    def nodo_solicitar_aclaracion(state: AgentState) -> AgentState:
        print("Ejecutando nodo 'solicitar_aclaracion'...")
        tri = state.get("triaje", {})
        campos = tri.get("campos_faltantes", [])
        detalle = f" Necesito que me indiques: {', '.join(campos)}." if campos else ""
        return {
            "respuesta": f"Necesito más información sobre tu consulta.{detalle}",
            "citaciones": [],
            "accion_final": "SOLICITAR_ACLARACION",
        }

    def nodo_escalar_a_soporte(state: AgentState) -> AgentState:
        print("Ejecutando nodo 'escalar_a_soporte'...")
        tri = state["triaje"]
        return {
            "respuesta": (
                f"Tu solicitud fue escalada al equipo de soporte "
                f"con urgencia {tri['urgencia']}. Un especialista te contactará pronto. "
                f"Caso: {state['pregunta']}."
            ),
            "citaciones": [],
            "accion_final": "ESCALAR_A_SOPORTE",
        }

    def arista_decision_triaje(state: AgentState) -> str:
        print("Decidiendo el flujo después del nodo 'triaje'...")
        decision = state["triaje"]["decision"]
        if decision == "RESPONDER_AUTOMATICO":
            return "responder_automatico"
        if decision == "ESCALAR_A_SOPORTE":
            return "escalar_a_soporte"
        return "solicitar_aclaracion"

    def arista_decision_responder_automatico(state: AgentState) -> str:
        print("Decidiendo el flujo después del nodo 'responder_automatico'...")
        if state.get("rag_exito"):
            print("RAG con éxito, finalizando el flujo.")
            return "end"
        print("RAG no encontró información suficiente, solicitando aclaración.")
        return "solicitar_aclaracion"

    workflow = StateGraph(AgentState)
    workflow.add_node("triaje", nodo_triaje)
    workflow.add_node("responder_automatico", nodo_responder_automatico)
    workflow.add_node("solicitar_aclaracion", nodo_solicitar_aclaracion)
    workflow.add_node("escalar_a_soporte", nodo_escalar_a_soporte)

    workflow.add_edge(START, "triaje")
    workflow.add_conditional_edges(
        "triaje",
        arista_decision_triaje,
        {
            "responder_automatico": "responder_automatico",
            "escalar_a_soporte": "escalar_a_soporte",
            "solicitar_aclaracion": "solicitar_aclaracion",
        },
    )
    workflow.add_conditional_edges(
        "responder_automatico",
        arista_decision_responder_automatico,
        {
            "end": END,
            "solicitar_aclaracion": "solicitar_aclaracion",
        },
    )
    workflow.add_edge("solicitar_aclaracion", END)
    workflow.add_edge("escalar_a_soporte", END)

    return workflow.compile()