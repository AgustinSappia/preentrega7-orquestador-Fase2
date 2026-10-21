"""
app/main.py
-----------
La API REST (FastAPI) que expone el orquestador multi-agente.

Endpoints:
    POST /tasks                -> crea un job, lo encola y devuelve el job_id YA (202)
    GET  /tasks/{job_id}       -> estado del job (PENDING/RUNNING/AWAITING_APPROVAL/DONE/REJECTED/FAILED)
    GET  /tasks                -> últimos jobs creados
    POST /tasks/{job_id}/approve  -> decisión humana (definido en app/hitl.py)
    GET  /reportes             -> reportes publicados (el efecto secundario que protege el HITL)
    GET  /health               -> estado de Redis y de la observabilidad

Cómo correrla:   uvicorn app.main:app --host 0.0.0.0 --port 8000
Documentación interactiva (Swagger) en http://localhost:8000/docs

¿Por qué la API no se bloquea?
    1. POST /tasks solo hace UNA escritura en Redis (async) y responde. El
       grafo se ejecuta después, como BackgroundTask.
    2. Todo lo que espera algo externo es `async` y se espera con `await`:
       Redis (`redis.asyncio`), el checkpointer (`AsyncRedisSaver`) y el
       grafo (`ainvoke`, que a su vez llama al LLM de forma asíncrona).
       Mientras una tarea espera al LLM, el event loop atiende otros
       pedidos. Ninguna función hace una llamada SÍNCRONA bloqueante.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.dependencies import obtener_grafo, obtener_jobs
from app.graph import checkpointer_redis, construir_grafo
from app.hitl import router as router_hitl
from app.jobs import Job, JobStore
from app.observability import cerrar_observabilidad, inicializar_observabilidad
from app.worker import ejecutar_job

load_dotenv(override=False)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Ciclo de vida: lo que se crea al ARRANCAR y se cierra al APAGAR la API
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Todo lo que está antes del `yield` corre una sola vez al arrancar;
    lo que está después, una sola vez al apagar."""
    app.state.observabilidad = inicializar_observabilidad()

    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    await redis.ping()  # si Redis no está levantado, la API falla al arrancar (mejor que fallar en cada pedido)
    jobs = JobStore(redis)

    async with checkpointer_redis(REDIS_URL) as checkpointer:
        app.state.jobs = jobs
        # Un solo grafo compilado para toda la API: cada job usa su propio
        # thread_id, así que los jobs concurrentes no se mezclan.
        app.state.grafo = construir_grafo(checkpointer=checkpointer, publicar=jobs.publicar_reporte)
        logger.info("API lista. Redis en %s", REDIS_URL)
        yield

    await redis.aclose()
    cerrar_observabilidad()


# ---------------------------------------------------------------------------
# Modelos de entrada/salida
# ---------------------------------------------------------------------------
class TaskCreate(BaseModel):
    topic: str = Field(min_length=2, max_length=200, description="Tema de noticias de IA a investigar.", examples=["regulación"])
    forzar_aprobacion: bool = Field(
        default=False, description="Si es true, la publicación SIEMPRE pasa por aprobación humana."
    )


class TaskCreada(BaseModel):
    job_id: str
    status: str
    links: dict[str, str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def crear_app(lifespan_fn=lifespan) -> FastAPI:
    """Fábrica de la app. Los tests la crean con `lifespan_fn=None` y le
    cargan dependencias falsas en `app.state`."""
    app = FastAPI(
        title="Orquestador multi-agente de noticias de IA",
        description="Pre-entrega 7: API asíncrona + Redis + LangSmith + Human-in-the-loop.",
        version="1.0.0",
        lifespan=lifespan_fn,
    )
    app.include_router(router_hitl)

    @app.post("/tasks", response_model=TaskCreada, status_code=status.HTTP_202_ACCEPTED)
    async def crear_tarea(
        pedido: TaskCreate,
        background_tasks: BackgroundTasks,
        jobs: JobStore = Depends(obtener_jobs),
        grafo: CompiledStateGraph = Depends(obtener_grafo),
    ) -> TaskCreada:
        """Encola la tarea y devuelve el job_id de inmediato (202 Accepted:
        "lo recibí, todavía no lo terminé")."""
        job = await jobs.crear(pedido.topic, forzar_aprobacion=pedido.forzar_aprobacion)
        background_tasks.add_task(ejecutar_job, job.job_id, grafo=grafo, jobs=jobs)
        return TaskCreada(
            job_id=job.job_id,
            status=job.status.value,
            links={"estado": f"/tasks/{job.job_id}", "aprobar": f"/tasks/{job.job_id}/approve"},
        )

    @app.get("/tasks/{job_id}", response_model=Job)
    async def estado_tarea(job_id: str, jobs: JobStore = Depends(obtener_jobs)) -> Job:
        job = await jobs.obtener(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
        return job

    @app.get("/tasks", response_model=list[Job])
    async def listar_tareas(limite: int = 20, jobs: JobStore = Depends(obtener_jobs)) -> list[Job]:
        return await jobs.listar(limite=min(max(limite, 1), 100))

    @app.get("/reportes")
    async def reportes(limite: int = 20, jobs: JobStore = Depends(obtener_jobs)) -> list[dict]:
        return await jobs.reportes_publicados(limite=min(max(limite, 1), 100))

    @app.get("/health")
    async def health(request: Request, jobs: JobStore = Depends(obtener_jobs)) -> dict:
        try:
            redis_ok = bool(await jobs.redis.ping())
        except Exception:  # noqa: BLE001
            redis_ok = False
        observabilidad: Optional[dict] = getattr(request.app.state, "observabilidad", None)
        return {"status": "ok" if redis_ok else "degradado", "redis": redis_ok, "observabilidad": observabilidad}

    return app


app = crear_app()
