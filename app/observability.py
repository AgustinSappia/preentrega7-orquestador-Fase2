"""
app/observability.py
--------------------
Capa de observabilidad con LangSmith.

¿Qué es una "traza"? Es el registro completo de UNA ejecución: un árbol
donde cada nodo del grafo, cada agente y cada llamada al LLM aparece como
un "span" (un tramo) con su entrada, su salida, cuánto tardó y cuántos
tokens consumió. En el dashboard de LangSmith eso permite responder lo
que pide la consigna: dónde se concentra la latencia, qué nodo consume
más tokens, cuánto cuesta cada ejecución y cuál es la latencia p95.

¿Cómo se activa? LangChain/LangGraph ya traen la instrumentación
incorporada: si existen estas variables de entorno, CADA `ainvoke` del
grafo se envía solo a LangSmith, sin tocar el código de los nodos:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=preentrega7-orquestador

Además, las funciones de los especialistas (`investigar`, `analizar`)
están decoradas con `@traceable`, para que aparezcan con nombre propio
dentro del árbol de la traza.

Lo que agrega ESTE archivo:
1. `inicializar_observabilidad()` + `verificar_conexion()`: al arrancar la
   API revisan que la key exista, que no sea el texto de ejemplo de
   .env.example y que LangSmith la acepte de verdad. Si algo falla, apagan
   el tracing y GET /health dice por qué, en vez de que las trazas se
   pierdan en silencio.
2. `config_de_ejecucion()`: la "etiqueta" que se le pega a cada ejecución
   (job_id, etapa, modelo) para poder filtrar y agrupar en el dashboard.
3. `cerrar_observabilidad()`: espera a que se terminen de enviar las
   trazas pendientes antes de apagar la API.
"""

import asyncio
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger("observability")

PROYECTO_POR_DEFECTO = "preentrega7-orquestador"


def _tracing_pedido() -> bool:
    # LANGSMITH_TRACING es el nombre actual; LANGCHAIN_TRACING_V2 el viejo.
    valor = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return valor.strip().lower() in {"1", "true", "yes", "si", "sí"}


def _parece_valor_de_ejemplo(api_key: str) -> bool:
    """Detecta si la key es el texto de ejemplo de .env.example ("lsv2_...")
    en vez de una key real. Sin este chequeo, copiar .env.example sin
    completar la key dejaba el tracing "activo" mientras LangSmith
    rechazaba todas las trazas en silencio."""
    return "..." in api_key or len(api_key) < 20


def _desactivar_tracing() -> None:
    """Apaga el tracing para todo el proceso."""
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        import langsmith
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()  # LangSmith guarda en caché las variables de entorno
        langsmith.configure(enabled=False)
    except Exception:  # noqa: BLE001
        logger.exception("No se pudo desactivar el tracing de LangSmith")


def inicializar_observabilidad() -> dict[str, Any]:
    """Se llama una vez al arrancar la API. Revisa la configuración (sin
    conectarse a nada) y devuelve un resumen que expone GET /health."""
    if not _tracing_pedido():
        logger.info("LangSmith desactivado (LANGSMITH_TRACING no está en true).")
        return {"plataforma": "langsmith", "activo": False, "motivo": "LANGSMITH_TRACING no está en true"}

    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not api_key:
        _desactivar_tracing()
        logger.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: se desactiva el tracing.")
        return {"plataforma": "langsmith", "activo": False, "motivo": "falta LANGSMITH_API_KEY"}

    if _parece_valor_de_ejemplo(api_key):
        _desactivar_tracing()
        logger.warning("LANGSMITH_API_KEY tiene el valor de ejemplo de .env.example: se desactiva el tracing.")
        return {
            "plataforma": "langsmith",
            "activo": False,
            "motivo": "LANGSMITH_API_KEY tiene el valor de ejemplo de .env.example (lsv2_...): poné tu key real",
        }

    proyecto = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or PROYECTO_POR_DEFECTO
    os.environ["LANGSMITH_PROJECT"] = proyecto
    return {"plataforma": "langsmith", "activo": True, "proyecto": proyecto, "conexion": "sin verificar"}


def _consultar_langsmith(cliente) -> None:
    """Una consulta mínima que requiere una key válida."""
    next(iter(cliente.list_projects(limit=1)), None)


async def verificar_conexion(
    estado: dict[str, Any], crear_cliente: Optional[Callable[[], Any]] = None, timeout: float = 8.0
) -> dict[str, Any]:
    """Confirma, al arrancar, que LangSmith ACEPTA la API key.

    - Key rechazada (401/403): se apaga el tracing y /health lo informa.
    - LangSmith no responde (red, timeout): se deja el tracing activo, pero
      /health avisa que la conexión no se pudo verificar.
    - Todo bien: `conexion: "ok"`.

    La consulta del cliente de LangSmith es síncrona, así que se corre en
    otro thread (`asyncio.to_thread`) para no bloquear el event loop.
    """
    if not estado.get("activo"):
        return estado

    if crear_cliente is None:
        from langsmith import Client

        crear_cliente = Client

    from langsmith.utils import LangSmithAuthError

    estado = dict(estado)
    try:
        await asyncio.wait_for(asyncio.to_thread(_consultar_langsmith, crear_cliente()), timeout=timeout)
    except LangSmithAuthError:
        _desactivar_tracing()
        logger.error("LangSmith rechazó LANGSMITH_API_KEY: se desactiva el tracing. Revisá la key en el .env.")
        estado.update(activo=False, conexion="rechazada", motivo="LangSmith rechazó LANGSMITH_API_KEY (401/403)")
        return estado
    except Exception as exc:  # noqa: BLE001 -- red caída, timeout, etc.
        logger.warning("No se pudo verificar la conexión con LangSmith: %s", exc)
        estado["conexion"] = f"sin verificar ({type(exc).__name__})"
        return estado

    estado["conexion"] = "ok"
    logger.info("LangSmith activo y verificado: las trazas se envían al proyecto '%s'.", estado["proyecto"])
    return estado


def config_de_ejecucion(job_id: str, etapa: str) -> dict[str, Any]:
    """Config de LangChain/LangGraph para UNA invocación del grafo.

    - `configurable.thread_id`: le dice al checkpointer de Redis en qué
      "hilo" guardar/leer los checkpoints. Usar el job_id hace que la
      reanudación (después de aprobar) continúe exactamente ese job.
    - `run_name`, `tags`, `metadata`: no cambian la ejecución; son
      etiquetas que viajan a LangSmith. Con `metadata.job_id` se puede
      buscar la traza de un job puntual, y con `metadata.thread_id`
      LangSmith agrupa en un mismo "Thread" la ejecución inicial y la
      reanudación posterior a la aprobación humana.
    """
    proveedor = os.getenv("LLM_PROVIDER", "openai")
    modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if proveedor == "openai" else os.getenv(
        "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
    )
    return {
        "configurable": {"thread_id": job_id},
        "run_name": f"orquestador-{etapa}",
        "tags": ["preentrega7", "orquestador", etapa],
        "metadata": {
            "job_id": job_id,
            "thread_id": job_id,
            "etapa": etapa,
            "llm_provider": proveedor,
            "llm_modelo": modelo,
        },
    }


def cerrar_observabilidad() -> None:
    """Las trazas se envían en segundo plano (en lotes). Al apagar la API
    hay que esperar a que salgan las últimas, o se pierden."""
    if not _tracing_pedido():
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:  # noqa: BLE001
        logger.exception("No se pudieron vaciar las trazas pendientes de LangSmith")
