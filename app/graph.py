"""
app/graph.py
------------
El orquestador multi-agente de la Pre-entrega 6 (Supervisor + Investigador
+ Analista), refactorizado para producción:

1. **Checkpointer de Redis** (`AsyncRedisSaver`): después de cada nodo,
   LangGraph guarda una "foto" del estado en Redis, identificada por el
   `thread_id` (que en la API es el `job_id`). Eso permite pausar el grafo
   (Human-in-the-loop) y reanudarlo minutos u horas después -- incluso si
   la API se reinició en el medio.
2. **Punto de interrupción crítico**: antes de publicar el reporte (una
   acción con efectos secundarios), el grafo pasa por el nodo
   "AprobacionHumana" (ver app/hitl.py).

Flujo completo:

    INICIO -> Supervisor --(Investigador)--> Investigador --> Supervisor
                         --(Analista)------> Analista ------> Supervisor
                         --(FINALIZAR, hay análisis suficiente)--> AprobacionHumana
                         --(FINALIZAR, sin análisis útil)--------> FIN
    AprobacionHumana --(aprobado)--> Publicador --> FIN
                     --(rechazado)-> FIN

Igual que en la Pre-entrega 6, todo lo "externo" se inyecta en
`construir_grafo()` (el LLM, la función de decisión del Supervisor, el
checkpointer y la función que publica) para poder testear el grafo
completo sin API keys ni Redis real.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agents.analyst_agent import analizar
from app.agents.research_agent import investigar
from app.hitl import nodo_aprobacion_humana, rutear_tras_aprobacion
from app.state import (
    MAX_PASOS,
    TIPOS_SERIALIZABLES,
    TIPOS_SERIALIZABLES_JSON,
    AgentState,
    ReportePublicado,
)

load_dotenv(override=False)

RECURSION_LIMIT = 50  # red de seguridad adicional, por detrás del contador `pasos`


# ---------------------------------------------------------------------------
# Decisión del Supervisor (igual que en la Pre-entrega 6)
# ---------------------------------------------------------------------------
class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor: el `Literal` garantiza que el
    LLM solo pueda devolver uno de estos tres valores."""

    next: Literal["Investigador", "Analista", "FINALIZAR"] = Field(
        description="El siguiente agente que debe actuar, o FINALIZAR si ya se puede responder."
    )


PROMPT_SUPERVISOR = (
    "Eres el supervisor de un sistema de investigación y análisis de "
    "noticias de Inteligencia Artificial. Coordinás a dos especialistas: "
    "el Investigador (busca noticias sobre un tema puntual) y el Analista "
    "(evalúa el sentimiento de las noticias que ya se encontraron). Mirá "
    "el estado actual del proceso y decidí quién debe actuar a "
    "continuación, o si ya se puede FINALIZAR.\n\n"
    "Reglas para decidir:\n"
    "- Si todavía no hay ninguna investigación hecha, mandá al "
    "Investigador.\n"
    "- Si ya hay una investigación (SIN IMPORTAR si encontró muchas, "
    "pocas o CERO noticias) pero todavía no hay un análisis sobre ella, "
    "mandá al Analista.\n"
    "- Si el análisis dice que fue insuficiente (no había noticias para "
    "analizar), mandá de nuevo al Investigador para que intente con un "
    "término de búsqueda más amplio -- esto es un refinamiento, no un "
    "error.\n"
    "- Si ya hay un análisis marcado como suficiente, FINALIZAR: la tarea "
    "está completa.\n"
)


def _resumen_estado(state: AgentState) -> str:
    """Resumen COMPACTO del estado para el Supervisor (no el historial
    completo de mensajes): evita la "Contaminación de Contexto"."""
    lineas = [f"Tema solicitado: {state['topic']}", f"Paso número: {state['pasos']}"]

    investigacion = state.get("investigacion")
    if investigacion is None:
        lineas.append("Investigación: todavía no se hizo.")
    else:
        lineas.append(
            f"Investigación: se buscó '{investigacion.tema_buscado}', "
            f"se encontraron {len(investigacion.noticias)} noticia(s)."
        )

    analisis = state.get("analisis")
    if analisis is None:
        lineas.append("Análisis: todavía no se hizo.")
    else:
        lineas.append(
            f"Análisis: sentimiento_general='{analisis.sentimiento_general}', "
            f"score_promedio={analisis.score_promedio}, suficiente={analisis.suficiente}."
        )

    return "\n".join(lineas)


DecisorType = Callable[[AgentState], Awaitable[str]]
PublicadorType = Callable[[ReportePublicado], Awaitable[None]]


def _construir_decisor_default(modelo) -> DecisorType:
    """Decisor de producción: las dos transiciones deterministas se
    resuelven en código y el LLM solo decide la que requiere criterio
    (refinar vs. finalizar). Ver README de la Pre-entrega 6."""
    llm_ruteo = modelo.with_structured_output(DecisionSupervisor)

    async def decidir(state: AgentState) -> str:
        if state.get("investigacion") is None:
            return "Investigador"
        if state.get("analisis") is None:
            return "Analista"

        mensajes = [
            SystemMessage(content=PROMPT_SUPERVISOR),
            HumanMessage(content=_resumen_estado(state)),
        ]
        decision = await llm_ruteo.ainvoke(mensajes)
        return decision.next

    return decidir


def _construir_modelo():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0
        )

    raise ValueError(f"Proveedor no soportado: {provider}")


async def _publicar_en_consola(reporte: ReportePublicado) -> None:
    """Publicador por defecto (si no se inyecta ninguno): solo imprime.
    En la API real se inyecta uno que escribe en Redis (ver app/jobs.py)."""
    print(f"[Publicador] Reporte publicado: {reporte.model_dump()}")


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------
def construir_grafo(
    modelo=None,
    decidir_siguiente_paso: Optional[DecisorType] = None,
    max_pasos: int = MAX_PASOS,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    publicar: Optional[PublicadorType] = None,
) -> CompiledStateGraph:
    """Arma y compila el StateGraph del orquestador.

    Args:
        modelo: el LLM de los especialistas (en tests, un modelo falso).
        decidir_siguiente_paso: la función de decisión del Supervisor (en
            tests, una función Python determinista).
        max_pasos: techo anti "Supervisor Infinito".
        checkpointer: dónde se guardan los checkpoints. En la API es un
            `AsyncRedisSaver`; en los tests, un `InMemorySaver`. Es
            OBLIGATORIO para que funcione el `interrupt()` del HITL.
        publicar: la acción con efectos secundarios que protege el HITL.
    """
    llm = modelo or _construir_modelo()
    decidir = decidir_siguiente_paso or _construir_decisor_default(llm)
    publicar = publicar or _publicar_en_consola

    async def supervisor_node(state: AgentState) -> dict:
        pasos = state.get("pasos", 0) + 1

        # Techo de seguridad contra el "Supervisor Infinito".
        if pasos > max_pasos:
            return {"next_agent": "FINALIZAR", "task_completed": True, "pasos": pasos}

        siguiente = await decidir(state)
        return {
            "next_agent": siguiente,
            "task_completed": siguiente == "FINALIZAR",
            "pasos": pasos,
        }

    async def investigador_node(state: AgentState) -> dict:
        analisis_previo = state.get("analisis")

        # Refinamiento: si el análisis anterior fue insuficiente, se amplía
        # la búsqueda a un término genérico en vez de repetir la misma.
        if analisis_previo is not None and not analisis_previo.suficiente:
            termino = "ia"
        else:
            termino = state["topic"]

        resultado = await investigar(llm, termino)
        # Una investigación nueva invalida cualquier análisis anterior.
        return {"investigacion": resultado, "analisis": None}

    async def analista_node(state: AgentState) -> dict:
        resultado = await analizar(llm, state["investigacion"])
        return {"analisis": resultado}

    async def publicador_node(state: AgentState) -> dict:
        """La acción "crítica" con efectos secundarios. Solo se llega acá
        si el nodo AprobacionHumana dejó `aprobacion.aprobado = True`."""
        analisis = state["analisis"]
        reporte = ReportePublicado(
            topic=state["topic"],
            sentimiento_general=analisis.sentimiento_general,
            score_promedio=analisis.score_promedio,
            cantidad_noticias=len(analisis.detalle),
            publicado_en=datetime.now(timezone.utc).isoformat(),
        )
        await publicar(reporte)
        return {"reporte": reporte}

    def rutear_desde_supervisor(state: AgentState) -> str:
        """Única arista condicional del Supervisor. Novedad de la
        Pre-entrega 7: FINALIZAR ya no va directo al FIN si hay un
        análisis útil -- primero pasa por la aprobación (y publicación)."""
        siguiente = state["next_agent"]
        if siguiente != "FINALIZAR":
            return siguiente

        analisis = state.get("analisis")
        if analisis is not None and analisis.suficiente:
            return "AprobacionHumana"
        # Se cortó por el contador de pasos sin un análisis útil: no hay
        # nada que publicar.
        return "FIN"

    grafo = StateGraph(AgentState)
    grafo.add_node("Supervisor", supervisor_node)
    grafo.add_node("Investigador", investigador_node)
    grafo.add_node("Analista", analista_node)
    grafo.add_node("AprobacionHumana", nodo_aprobacion_humana)
    grafo.add_node("Publicador", publicador_node)

    grafo.set_entry_point("Supervisor")
    grafo.add_conditional_edges(
        "Supervisor",
        rutear_desde_supervisor,
        {
            "Investigador": "Investigador",
            "Analista": "Analista",
            "AprobacionHumana": "AprobacionHumana",
            "FIN": END,
        },
    )
    grafo.add_edge("Investigador", "Supervisor")
    grafo.add_edge("Analista", "Supervisor")
    grafo.add_conditional_edges(
        "AprobacionHumana",
        rutear_tras_aprobacion,
        {"Publicador": "Publicador", "FIN": END},
    )
    grafo.add_edge("Publicador", END)

    return grafo.compile(checkpointer=checkpointer)


def estado_inicial(topic: str, forzar_aprobacion: bool = False) -> AgentState:
    """Estado inicial completo (todos los campos presentes desde el
    arranque, ninguno implícito)."""
    return {
        "messages": [],
        "topic": topic,
        "next_agent": "",
        "task_completed": False,
        "pasos": 0,
        "investigacion": None,
        "analisis": None,
        "forzar_aprobacion": forzar_aprobacion,
        "aprobacion": None,
        "reporte": None,
    }


# ---------------------------------------------------------------------------
# Checkpointer de Redis
# ---------------------------------------------------------------------------
def serializador_redis():
    """Serializador del checkpointer de Redis con la "lista blanca" de
    clases propias que se pueden reconstruir al leer un checkpoint.

    Se configuran LOS DOS formatos, porque el checkpointer de Redis usa
    ambos: msgpack para algunos datos y JSON (RedisJSON) para el estado de
    los canales. Con solo `allowed_msgpack_modules`, al reanudar un job
    después de la aprobación humana los artefactos volvían como `dict`.
    (Este error apareció en la primera corrida real con Redis 8: el
    checkpointer en memoria de los tests usa solo msgpack y no lo mostraba.
    Ahora lo cubre tests/test_checkpointer_redis.py.)
    """
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    return JsonPlusRedisSerializer(
        allowed_msgpack_modules=TIPOS_SERIALIZABLES,
        allowed_json_modules=TIPOS_SERIALIZABLES_JSON,
    )


@asynccontextmanager
async def checkpointer_redis(redis_url: str, ttl_minutos: int = 24 * 60) -> AsyncIterator[BaseCheckpointSaver]:
    """Abre un `AsyncRedisSaver` listo para usar y lo cierra al salir.

    - Necesita Redis 8 (o Redis Stack): usa los módulos de búsqueda y JSON
      de Redis para indexar los checkpoints. El docker-compose.yml del
      proyecto ya levanta la imagen `redis:8`.
    - `ttl`: los checkpoints expiran solos después de `ttl_minutos` (por
      defecto 24 h), para que Redis no crezca indefinidamente.
    - `serde`: se reemplaza el serializador por uno con la "lista blanca"
      de clases propias que se pueden reconstruir (ver state.py).
    """
    # Import local: así los tests (que usan InMemorySaver) no necesitan
    # cargar la librería de Redis.
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": ttl_minutos, "refresh_on_read": True}
    ) as saver:
        saver.serde = serializador_redis()
        await saver.asetup()
        yield saver
