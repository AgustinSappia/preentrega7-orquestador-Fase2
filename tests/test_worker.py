"""
tests/test_worker.py
--------------------
Tests del worker (app/worker.py): que cada camino deje el job en el
estado correcto en Redis -- en especial, que CUALQUIER falla termine en
FAILED (si no, el cliente quedaría haciendo polling para siempre).
"""
import asyncio

import pytest

from app import worker
from app.graph import construir_grafo
from app.jobs import EstadoJob, JobStore
from app.state import DecisionHumana
from tests.fakes import ModeloFalso, decisor_de_referencia, respuestas_investigar_y_analizar


def _grafo(tema: str, jobs: JobStore, checkpointer):
    return construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar(tema)),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
        publicar=jobs.publicar_reporte,  # el publicador REAL (sobre fakeredis)
    )


class GrafoQueExplota:
    """Doble de prueba: simula un grafo que falla adentro (p. ej. el
    proveedor del LLM devolvió un error)."""

    async def ainvoke(self, entrada, config=None):
        raise RuntimeError("el proveedor del LLM no respondió")


class GrafoLento:
    async def ainvoke(self, entrada, config=None):
        await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Caminos exitosos
# ---------------------------------------------------------------------------
async def test_job_no_critico_termina_en_done_y_publica(jobs: JobStore, checkpointer):
    job = await jobs.crear("modelos")
    await worker.ejecutar_job(job.job_id, grafo=_grafo("modelos", jobs, checkpointer), jobs=jobs)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.DONE
    assert final.resultado["analisis"]["sentimiento_general"] == "positivo"
    assert final.resultado["reporte_publicado"] is not None
    assert final.duracion_segundos is not None
    assert len(await jobs.reportes_publicados()) == 1


async def test_job_critico_queda_esperando_y_al_aprobar_termina(jobs: JobStore, checkpointer):
    grafo = _grafo("regulación", jobs, checkpointer)
    job = await jobs.crear("regulación")
    await worker.ejecutar_job(job.job_id, grafo=grafo, jobs=jobs)

    pausado = await jobs.obtener(job.job_id)
    assert pausado.status == EstadoJob.AWAITING_APPROVAL
    assert pausado.aprobacion_pendiente["propuesta"]["sentimiento_general"] == "negativo"
    assert await jobs.reportes_publicados() == []  # nada publicado todavía

    # Lo mismo que hace el endpoint POST /approve:
    await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING)
    await worker.reanudar_job(job.job_id, DecisionHumana(aprobado=True, revisor="rayo"), grafo=grafo, jobs=jobs)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.DONE
    assert final.aprobacion_pendiente is None
    assert final.resultado["aprobacion"]["revisor"] == "rayo"
    assert len(await jobs.reportes_publicados()) == 1


async def test_job_critico_rechazado_termina_en_rejected(jobs: JobStore, checkpointer):
    grafo = _grafo("regulación", jobs, checkpointer)
    job = await jobs.crear("regulación")
    await worker.ejecutar_job(job.job_id, grafo=grafo, jobs=jobs)
    await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING)
    await worker.reanudar_job(job.job_id, DecisionHumana(aprobado=False), grafo=grafo, jobs=jobs)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.REJECTED
    assert final.resultado["reporte_publicado"] is None
    assert await jobs.reportes_publicados() == []


# ---------------------------------------------------------------------------
# Manejo de errores: SIEMPRE termina en FAILED
# ---------------------------------------------------------------------------
async def test_excepcion_en_el_grafo_marca_failed(jobs: JobStore):
    job = await jobs.crear("modelos")
    # No debe propagar la excepción (si lo hiciera, se perdería dentro de
    # la background task).
    await worker.ejecutar_job(job.job_id, grafo=GrafoQueExplota(), jobs=jobs)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.FAILED
    assert "RuntimeError" in final.error
    assert "no respondió" in final.error
    assert final.es_final  # el cliente deja de hacer polling


async def test_timeout_marca_failed(jobs: JobStore, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker, "TIMEOUT_JOB_SEGUNDOS", 0.05)
    job = await jobs.crear("modelos")
    await worker.ejecutar_job(job.job_id, grafo=GrafoLento(), jobs=jobs)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.FAILED
    assert final.error.startswith("Timeout")


async def test_excepcion_al_reanudar_marca_failed(jobs: JobStore):
    job = await jobs.crear("regulación")
    await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)
    await worker.reanudar_job(job.job_id, DecisionHumana(aprobado=True), grafo=GrafoQueExplota(), jobs=jobs)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.FAILED


def test_config_de_ejecucion_lleva_el_job_id():
    from app.observability import config_de_ejecucion

    config = config_de_ejecucion("abc-123", etapa="ejecucion")
    assert config["configurable"]["thread_id"] == "abc-123"
    assert config["metadata"]["job_id"] == "abc-123"
    assert "ejecucion" in config["tags"]


async def test_cinco_jobs_concurrentes_con_el_mismo_grafo_no_se_mezclan(jobs: JobStore, checkpointer):
    """Un solo grafo compilado + un solo checkpointer (como en la API) y 5
    jobs corriendo A LA VEZ: cada uno tiene que terminar con SU tema, gracias
    a que cada job usa su propio thread_id."""
    from tests.fakes import ModeloSegunContexto

    grafo = construir_grafo(
        modelo=ModeloSegunContexto(),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
        publicar=jobs.publicar_reporte,
    )
    temas = ["modelos", "regulación", "robótica", "adopción", "agentes"]
    creados = [await jobs.crear(t) for t in temas]

    await asyncio.gather(*[worker.ejecutar_job(j.job_id, grafo=grafo, jobs=jobs) for j in creados])

    finales = {j.topic: await jobs.obtener(j.job_id) for j in creados}
    for tema, job in finales.items():
        if job.status == EstadoJob.DONE:
            assert job.resultado["topic"] == tema
        else:
            assert job.aprobacion_pendiente["propuesta"]["topic"] == tema
    # regulación y agentes son negativos -> esperan aprobación; el resto se publica.
    assert {t for t, j in finales.items() if j.status == EstadoJob.AWAITING_APPROVAL} == {"regulación", "agentes"}
    assert {t for t, j in finales.items() if j.status == EstadoJob.DONE} == {"modelos", "robótica", "adopción"}
    assert len(await jobs.reportes_publicados()) == 3
