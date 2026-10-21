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
1. `inicializar_observabilidad()`: valida la configuración al arrancar la
   API (si falta la API key, apaga el tracing con un aviso claro en vez
   de llenar el log de errores).
2. `config_de_ejecucion()`: la "etiqueta" que se le pega a cada ejecución
   (job_id, etapa, modelo) para poder filtrar y agrupar en el dashboard.
3. `cerrar_observabilidad()`: espera a que se terminen de enviar las
   trazas pendientes antes de apagar la API.
"""

import logging
import os
from typing import Any

logger = logging.getLogger("observability")

PROYECTO_POR_DEFECTO = "preentrega7-orquestador"


def _tracing_pedido() -> bool:
    # LANGSMITH_TRACING es el nombre actual; LANGCHAIN_TRACING_V2 el viejo.
    valor = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return valor.strip().lower() in {"1", "true", "yes", "si", "sí"}


def inicializar_observabilidad() -> dict[str, Any]:
    """Se llama una vez al arrancar la API. Devuelve un resumen del estado
    de la observabilidad (lo expone GET /health)."""
    if not _tracing_pedido():
        logger.info("LangSmith desactivado (LANGSMITH_TRACING no está en true).")
        return {"plataforma": "langsmith", "activo": False, "motivo": "LANGSMITH_TRACING no está en true"}

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        # Mejor apagarlo explícitamente que dejar que cada traza falle.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: se desactiva el tracing.")
        return {"plataforma": "langsmith", "activo": False, "motivo": "falta LANGSMITH_API_KEY"}

    proyecto = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or PROYECTO_POR_DEFECTO
    os.environ["LANGSMITH_PROJECT"] = proyecto
    logger.info("LangSmith activo: las trazas se envían al proyecto '%s'.", proyecto)
    return {"plataforma": "langsmith", "activo": True, "proyecto": proyecto}


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
