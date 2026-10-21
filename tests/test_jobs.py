"""
tests/test_jobs.py
------------------
Tests del estado de los jobs en Redis (app/jobs.py), con fakeredis.
"""
import asyncio

from app.jobs import EstadoJob, JobStore
from app.state import ReportePublicado


async def test_crear_y_obtener_job(jobs: JobStore):
    job = await jobs.crear("modelos")
    assert job.status == EstadoJob.PENDING

    leido = await jobs.obtener(job.job_id)
    assert leido is not None
    assert leido.topic == "modelos"
    assert leido.status == EstadoJob.PENDING


async def test_obtener_job_inexistente_devuelve_none(jobs: JobStore):
    assert await jobs.obtener("no-existe") is None


async def test_el_job_queda_guardado_con_vencimiento(jobs: JobStore, redis_falso):
    job = await jobs.crear("modelos")
    ttl = await redis_falso.ttl(f"job:{job.job_id}")
    assert 0 < ttl <= jobs.ttl_segundos


async def test_ciclo_de_vida_registra_tiempos(jobs: JobStore):
    job = await jobs.crear("modelos")
    corriendo = await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)
    assert corriendo.iniciado_en is not None
    assert corriendo.finalizado_en is None

    terminado = await jobs.actualizar(job.job_id, status=EstadoJob.DONE, resultado={"ok": True})
    assert terminado.finalizado_en is not None
    assert terminado.duracion_segundos is not None and terminado.duracion_segundos >= 0
    assert terminado.es_final


async def test_transicionar_solo_desde_el_estado_esperado(jobs: JobStore):
    job = await jobs.crear("regulación")
    # Desde PENDING no se puede "aprobar".
    assert await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING) is None

    await jobs.actualizar(job.job_id, status=EstadoJob.AWAITING_APPROVAL)
    actualizado = await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING)
    assert actualizado is not None
    assert actualizado.status == EstadoJob.RUNNING


async def test_doble_aprobacion_simultanea_solo_gana_una(jobs: JobStore):
    """Dos pedidos de aprobación al mismo tiempo: gracias a WATCH/MULTI,
    exactamente uno logra pasar el job a RUNNING."""
    job = await jobs.crear("regulación")
    await jobs.actualizar(job.job_id, status=EstadoJob.AWAITING_APPROVAL)

    resultados = await asyncio.gather(
        jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING),
        jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING),
    )
    ganadores = [r for r in resultados if r is not None]
    assert len(ganadores) == 1


async def test_publicar_reporte_en_redis(jobs: JobStore):
    reporte = ReportePublicado(
        topic="modelos",
        sentimiento_general="positivo",
        score_promedio=0.8,
        cantidad_noticias=2,
        publicado_en="2026-10-21T00:00:00+00:00",
    )
    await jobs.publicar_reporte(reporte)
    publicados = await jobs.reportes_publicados()
    assert len(publicados) == 1
    assert publicados[0]["topic"] == "modelos"


async def test_listar_devuelve_los_mas_nuevos_primero(jobs: JobStore):
    primero = await jobs.crear("a")
    await asyncio.sleep(0.01)
    segundo = await jobs.crear("b")
    listados = await jobs.listar()
    assert [j.job_id for j in listados] == [segundo.job_id, primero.job_id]
