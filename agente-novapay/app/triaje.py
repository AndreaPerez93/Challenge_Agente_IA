"""
Clasifica la pregunta del colaborador o usuario en:
RESPONDER_AUTOMATICO / SOLICITAR_ACLARACION / ESCALAR_A_SOPORTE

Adaptado al contexto de un banco digital (NovaPay): las excepciones,
fraudes y solicitudes fuera de política nunca se resuelven solas,
siempre se escalan a un humano.
"""
from typing import Literal, List, Dict

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.rag import llm

PROMPT_TRIAJE = """
Eres el especialista en triaje de NovaPay, un banco digital.
Tu trabajo es clasificar el mensaje del usuario antes de decidir cómo responderlo.

Devuelve SÓLO un JSON con esta forma:
{
 "decision": "RESPONDER_AUTOMATICO" | "SOLICITAR_ACLARACION" | "ESCALAR_A_SOPORTE",
 "urgencia": "BAJA" | "MEDIANA" | "ALTA",
 "campos_faltantes": ["..."]
}

Reglas de decisión:
- RESPONDER_AUTOMATICO: preguntas sobre políticas, límites, tarifas, procesos o
  procedimientos ya documentados. Ejemplo: "¿cuánto cobra una transferencia SPEI?",
  "¿cuál es mi límite diario?".
- SOLICITAR_ACLARACION: mensajes ambiguos, incompletos o que no identifican
  claramente el tema. Ejemplo: "tengo un problema con mi cuenta", "no me deja hacer algo".
- ESCALAR_A_SOPORTE: siempre que se trate de fraude, cargos no reconocidos,
  solicitudes de excepción a límites o tarifas, cierre de cuenta, o cualquier caso
  que requiera intervención humana o aprobación. Si el usuario menciona la palabra
  "fraude" o "no reconozco este cargo", la urgencia debe ser "ALTA" sin excepción.

Analiza el mensaje y decide la acción más adecuada.
"""


class TriajeOut(BaseModel):
    decision: Literal["RESPONDER_AUTOMATICO", "SOLICITAR_ACLARACION", "ESCALAR_A_SOPORTE"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


chain_de_triaje = llm.with_structured_output(TriajeOut)


def triaje(mensaje: str) -> Dict:
    salida: TriajeOut = chain_de_triaje.invoke(
        [
            SystemMessage(content=PROMPT_TRIAJE),
            HumanMessage(content=mensaje),
        ]
    )
    return salida.model_dump()