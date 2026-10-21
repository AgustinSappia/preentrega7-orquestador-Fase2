"""
tests/fakes.py
---------------
Utilidades compartidas por toda la suite de tests: un modelo de chat
"falso" que no llama a ninguna API real, pero que SÍ es una instancia
genuina de `BaseChatModel` (requisito de `create_react_agent`, que
rechaza objetos duck-typed que no sean un `Runnable` real -- ver nota
más abajo).
"""
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class ModeloFalso(FakeMessagesListChatModel):
    """Subclase de un `BaseChatModel` real que devuelve, en orden, los
    mensajes de la lista `responses` que se le pasó al construirlo.

    Hace falta subclasearlo para sobreescribir `bind_tools`: la
    implementación por defecto de `BaseChatModel.bind_tools` levanta
    `NotImplementedError`, y `create_react_agent` siempre llama a
    `bind_tools` sobre el modelo que recibe. Como en estos tests nunca
    nos importa qué herramientas quedaron "bindeadas" (el modelo falso
    ya tiene sus respuestas guionadas de antemano), alcanza con
    devolver `self` sin hacer nada más.
    """

    def bind_tools(self, tools, **kwargs: Any):
        return self


# ---------------------------------------------------------------------------
# [Pre-entrega 7] Helpers para guionar corridas completas del grafo
# ---------------------------------------------------------------------------
from langchain_core.messages import AIMessage  # noqa: E402

from app.agents.research_agent import _NOTICIAS_IA  # noqa: E402


def noticias_que_matchean(termino: str) -> list[dict]:
    """Replica la búsqueda simulada del Investigador, para poder armar de
    antemano el argumento que el "LLM" le pasa al Analista."""
    termino = termino.lower()
    salida = []
    for n in _NOTICIAS_IA:
        texto = " ".join([n["titular"], n["resumen"], *n["etiquetas"]]).lower()
        if termino in texto:
            salida.append({k: n[k] for k in ("titular", "fuente", "fecha", "resumen")})
    return salida


def respuestas_investigar_y_analizar(tema: str) -> list[AIMessage]:
    """Las 4 respuestas que da el LLM falso en una corrida "feliz":
    tool-call de búsqueda, texto, tool-call de análisis, texto."""
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": tema, "maximo": 5}, "id": "c1"}],
        ),
        AIMessage(content="Encontré noticias."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "analizar_sentimiento_noticias",
                    "args": {"noticias": noticias_que_matchean(tema)},
                    "id": "c2",
                }
            ],
        ),
        AIMessage(content="Listo, analicé el sentimiento."),
    ]


async def decisor_de_referencia(state) -> str:
    """Implementación en Python puro de la lógica de ruteo del Supervisor."""
    if state.get("investigacion") is None:
        return "Investigador"
    if state.get("analisis") is None:
        return "Analista"
    if not state["analisis"].suficiente:
        return "Investigador"
    return "FINALIZAR"


# ---------------------------------------------------------------------------
# [Pre-entrega 7] LLM falso "sin memoria", seguro para tareas concurrentes
# ---------------------------------------------------------------------------
import json  # noqa: E402
import uuid  # noqa: E402

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ModeloSegunContexto(BaseChatModel):
    """A diferencia de `ModeloFalso` (que devuelve una lista de respuestas
    EN ORDEN), este decide qué responder mirando el último mensaje. Así
    varios jobs pueden usar el mismo modelo a la vez sin "robarse" las
    respuestas entre ellos -- igual que pasa con un LLM real."""

    @property
    def _llm_type(self) -> str:
        return "modelo-segun-contexto"

    def bind_tools(self, tools, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        ultimo = messages[-1]
        texto = ultimo.content if isinstance(ultimo.content, str) else ""

        if isinstance(ultimo, ToolMessage):
            respuesta = AIMessage(content="Listo.")
        elif texto.startswith("Buscá noticias sobre:"):
            tema = texto.split(":", 1)[1].strip()
            respuesta = AIMessage(
                content="",
                tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": tema}, "id": f"c-{uuid.uuid4().hex[:8]}"}],
            )
        elif "analizar_sentimiento_noticias" in texto and "\n" in texto:
            noticias = json.loads(texto.split("\n", 1)[1])
            respuesta = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analizar_sentimiento_noticias",
                        "args": {"noticias": noticias},
                        "id": f"c-{uuid.uuid4().hex[:8]}",
                    }
                ],
            )
        else:
            respuesta = AIMessage(content="No tengo nada que hacer.")
        return ChatResult(generations=[ChatGeneration(message=respuesta)])
