"""
app/worker.py
-------------
El "worker": la función que corre el grafo EN SEGUNDO PLANO y va
anotando en Redis cómo va el job.

¿Quién la llama? El endpoint POST /tasks la agenda con
`BackgroundTasks` de FastAPI. FastAPI primero RESPONDE al cliente (con el
job_id) y recién después ejecuta esta función, dentro del mismo event
loop. Como todo acá adentro es `async` (el grafo con `ainvoke`, Redis con
`redis.asyncio`), mientras el grafo espera la respuesta del LLM el event
loop queda libre para atender otros pedidos: la API nunca se bloquea.

Regla de oro de este archivo (uno de los "errores comunes" de la
consigna): NINGUNA excepción puede escaparse sin dejar el job en FAILED.
Una excepción que se pierde dentro de una background task no le llega a
nadie -- y el cliente quedaría haciendo polling de un job en RUNNING para
siempre. Por eso las dos funciones públicas envuelven TODO en
`try/except Exception` y, además, tienen un timeout.
"""

import asyncio
import logging
import os
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.graph import RECURSION_LIMIT, estado_inicial
from app.jobs import EstadoJob, JobStore
from app.observability import config_de_ejecucion
from app.state import DecisionHumana

logger = logging.getLogger("worker")

# Si el grafo tarda más que esto, se corta y el job queda FAILED (en vez
# de quedar colgado en RUNNING si, por ejemplo, el proveedor del LLM no
# responde nunca).
TIMEOUT_JOB_SEGUNDOS = float(os.getenv("TIMEOUT_JOB_SEGUNDOS", "300"))


async def ejecutar_job(job_id: str, *, grafo: CompiledStateGraph, jobs: JobStore) -> None:
    """Corre el grafo desde el principio para un job recién creado."""
    try:
        job = await jobs.actualizar(job_id, status=EstadoJob.RUNNING)
        entrada = estado_inicial(job.topic, forzar_aprobacion=job.forzar_aprobacion)
        await _correr_grafo(job_id, entrada, grafo=grafo, jobs=jobs, etapa="ejecucion")
    except Exception as exc:  # noqa: BLE001 -- a propósito: capturar TODO
        await _marcar_fallido(job_id, exc, jobs)


async def reanudar_job(
    job_id: str, decision: DecisionHumana, *, grafo: CompiledStateGraph, jobs: JobStore
) -> None:
    """Reanuda un grafo pausado en AprobacionHumana con la decisión humana.

    El endpoint de aprobación ya pasó el job de AWAITING_APPROVAL a RUNNING
    (de forma atómica) antes de agendar esta función."""
    try:
        await _correr_grafo(
            job_id, Command(resume=decision.model_dump()), grafo=grafo, jobs=jobs, etapa="reanudacion"
        )
    except Exception as exc:  # noqa: BLE001
        await _marcar_fallido(job_id, exc, jobs)


# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------
async def _correr_grafo(job_id: str, entrada: Any, *, grafo: CompiledStateGraph, jobs: JobStore, etapa: str) -> None:
    config = config_de_ejecucion(job_id, etapa=etapa)
    config["recursion_limit"] = RECURSION_LIMIT

    resultado = await asyncio.wait_for(grafo.ainvoke(entrada, config=config), timeout=TIMEOUT_JOB_SEGUNDOS)
    await _registrar_resultado(job_id, resultado, jobs)


async def _registrar_resultado(job_id: str, resultado: dict, jobs: JobStore) -> None:
    """Traduce lo que devolvió el grafo a un estado del job."""
    # Caso 1: el grafo se detuvo en un interrupt() -> espera aprobación.
    interrupciones = resultado.get("__interrupt__")
    if interrupciones:
        await jobs.actualizar(
            job_id,
            status=EstadoJob.AWAITING_APPROVAL,
            aprobacion_pendiente=interrupciones[0].value,
        )
        logger.info("Job %s pausado esperando aprobación humana", job_id)
        return

    # Caso 2: el grafo terminó.
    resumen = resumir_resultado(resultado)
    aprobacion = resultado.get("aprobacion")
    status = EstadoJob.REJECTED if aprobacion is not None and not aprobacion.aprobado else EstadoJob.DONE
    await jobs.actualizar(job_id, status=status, resultado=resumen, aprobacion_pendiente=None)
    logger.info("Job %s terminado con estado %s", job_id, status.value)


def resumir_resultado(resultado: dict) -> dict:
    """Arma un resumen JSON-serializable del estado final del grafo (los
    artefactos Pydantic se convierten con `model_dump(mode="json")`)."""

    def volcar(valor):
        return valor.model_dump(mode="json") if valor is not None else None

    investigacion = resultado.get("investigacion")
    return {
        "topic": resultado.get("topic"),
        "pasos": resultado.get("pasos"),
        "tema_buscado": investigacion.tema_buscado if investigacion else None,
        "cantidad_noticias": len(investigacion.noticias) if investigacion else 0,
        "analisis": volcar(resultado.get("analisis")),
        "aprobacion": volcar(resultado.get("aprobacion")),
        "reporte_publicado": volcar(resultado.get("reporte")),
    }


async def _marcar_fallido(job_id: str, exc: BaseException, jobs: JobStore) -> None:
    if isinstance(exc, asyncio.TimeoutError):
        mensaje = f"Timeout: el grafo tardó más de {TIMEOUT_JOB_SEGUNDOS:.0f} s"
    else:
        mensaje = f"{type(exc).__name__}: {exc}"
    logger.exception("Job %s falló: %s", job_id, mensaje)

    try:
        await jobs.actualizar(job_id, status=EstadoJob.FAILED, error=mensaje, aprobacion_pendiente=None)
    except Exception:  # noqa: BLE001
        # Si ni siquiera Redis responde, no hay a dónde anotar el error:
        # al menos queda en el log del servidor.
        logger.exception("No se pudo marcar el job %s como FAILED", job_id)
