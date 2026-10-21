"""
app/jobs.py
-----------
El "tablero de turnos": dónde vive el estado de cada job en Redis.

Ojo con la diferencia entre las dos cosas que guardamos en Redis:

- **Checkpoints de LangGraph** (los maneja `AsyncRedisSaver`, ver
  graph.py): la "foto" interna del grafo, nodo por nodo. Sirven para
  pausar/reanudar. El cliente de la API nunca los ve directamente.
- **Estado del job** (este archivo): un resumen pensado para el cliente
  de la API -- ¿en qué etapa está mi pedido?, ¿terminó?, ¿falló y por
  qué?, ¿está esperando aprobación? Es lo que devuelve GET /tasks/{id}.

Ciclo de vida de un job:

    PENDING ──> RUNNING ──┬──> DONE                (terminó y publicó / no había nada que publicar)
                          ├──> AWAITING_APPROVAL ──(POST /approve)──> RUNNING ──> DONE / REJECTED
                          └──> FAILED              (cualquier excepción o timeout en el worker)

Cada job se guarda como un JSON en la clave `job:{job_id}`, con un TTL
(vencimiento) para que Redis no crezca para siempre. Se usa
`redis.asyncio` (el cliente ASÍNCRONO de Redis): así ninguna lectura o
escritura bloquea el event loop de FastAPI.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.state import ReportePublicado

CLAVE_REPORTES_PUBLICADOS = "reportes:publicados"
CLAVE_INDICE_JOBS = "jobs:indice"


class EstadoJob(str, Enum):
    """Estados posibles de un job. Heredar de `str` hace que se guarden y
    se devuelvan en el JSON como texto plano ("RUNNING"), no como un
    objeto raro de Python."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


ESTADOS_FINALES = {EstadoJob.DONE, EstadoJob.REJECTED, EstadoJob.FAILED}


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job(BaseModel):
    """Lo que se guarda en Redis (y lo que ve el cliente en GET /tasks/{id})."""

    job_id: str
    status: EstadoJob
    topic: str
    forzar_aprobacion: bool = False

    creado_en: str = Field(default_factory=_ahora)
    actualizado_en: str = Field(default_factory=_ahora)
    iniciado_en: Optional[str] = None
    finalizado_en: Optional[str] = None
    duracion_segundos: Optional[float] = None  # desde que arrancó a correr hasta el estado final

    # Solo cuando status == AWAITING_APPROVAL: lo que el humano tiene que revisar.
    aprobacion_pendiente: Optional[dict[str, Any]] = None
    # Solo cuando status == DONE / REJECTED: el resumen del resultado.
    resultado: Optional[dict[str, Any]] = None
    # Solo cuando status == FAILED: qué salió mal (para que el cliente
    # deje de hacer polling y sepa por qué).
    error: Optional[str] = None

    @property
    def es_final(self) -> bool:
        return self.status in ESTADOS_FINALES


class JobStore:
    """Encapsula TODO el acceso a Redis para los jobs.

    El resto del código (API y worker) nunca habla con Redis directo: le
    pide cosas a esta clase. Eso permite testearla con `fakeredis` (un
    Redis falso en memoria) sin cambiar nada del código de producción.
    """

    def __init__(self, redis: Redis, ttl_segundos: int = 7 * 24 * 3600):
        self.redis = redis
        self.ttl_segundos = ttl_segundos

    @staticmethod
    def _clave(job_id: str) -> str:
        return f"job:{job_id}"

    # -- Lectura -------------------------------------------------------------
    async def obtener(self, job_id: str) -> Optional[Job]:
        crudo = await self.redis.get(self._clave(job_id))
        return Job.model_validate_json(crudo) if crudo else None

    async def listar(self, limite: int = 20) -> list[Job]:
        """Los últimos `limite` jobs creados (más nuevos primero)."""
        ids = await self.redis.zrevrange(CLAVE_INDICE_JOBS, 0, limite - 1)
        jobs = [await self.obtener(job_id) for job_id in ids]
        return [job for job in jobs if job is not None]

    # -- Escritura -----------------------------------------------------------
    async def crear(self, topic: str, forzar_aprobacion: bool = False) -> Job:
        job = Job(
            job_id=str(uuid.uuid4()),
            status=EstadoJob.PENDING,
            topic=topic,
            forzar_aprobacion=forzar_aprobacion,
        )
        await self._guardar(job)
        await self.redis.zadd(CLAVE_INDICE_JOBS, {job.job_id: time.time()})
        return job

    async def actualizar(self, job_id: str, **cambios: Any) -> Job:
        """Cambia campos de un job (usado por el worker, que es el único que
        escribe mientras el job corre)."""
        job = await self.obtener(job_id)
        if job is None:
            raise KeyError(f"No existe el job {job_id}")
        job = self._aplicar_cambios(job, cambios)
        await self._guardar(job)
        return job

    async def transicionar(
        self, job_id: str, desde: set[EstadoJob], hacia: EstadoJob, **cambios: Any
    ) -> Optional[Job]:
        """Cambia el estado SOLO si el estado actual está en `desde`, de
        forma atómica. Devuelve el job actualizado, o None si no se pudo.

        ¿Para qué? Si dos personas aprietan "aprobar" al mismo tiempo, sin
        esto los dos pedidos podrían ver AWAITING_APPROVAL y reanudar el
        grafo DOS veces (publicando el reporte dos veces). Con `WATCH`,
        Redis aborta la transacción si alguien más modificó la clave entre
        que la leímos y la escribimos: el segundo pedido "pierde" y recibe
        None. Esto se llama "concurrencia optimista".
        """
        clave = self._clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(clave)
                    crudo = await pipe.get(clave)
                    if not crudo:
                        return None
                    job = Job.model_validate_json(crudo)
                    if job.status not in desde:
                        return None
                    job = self._aplicar_cambios(job, {"status": hacia, **cambios})
                    pipe.multi()
                    pipe.set(clave, job.model_dump_json(), ex=self.ttl_segundos)
                    await pipe.execute()
                    return job
                except WatchError:
                    # Alguien modificó el job en el medio: se reintenta
                    # leyendo el estado nuevo (que probablemente ya no
                    # esté en `desde`, y entonces devuelve None).
                    continue

    async def publicar_reporte(self, reporte: ReportePublicado) -> None:
        """El efecto secundario que protege el Human-in-the-loop: agrega
        el reporte al "canal de publicación" (una lista de Redis)."""
        await self.redis.lpush(CLAVE_REPORTES_PUBLICADOS, reporte.model_dump_json())

    async def reportes_publicados(self, limite: int = 20) -> list[dict]:
        crudos = await self.redis.lrange(CLAVE_REPORTES_PUBLICADOS, 0, limite - 1)
        return [json.loads(c) for c in crudos]

    # -- Internos ------------------------------------------------------------
    async def _guardar(self, job: Job) -> None:
        await self.redis.set(self._clave(job.job_id), job.model_dump_json(), ex=self.ttl_segundos)

    @staticmethod
    def _aplicar_cambios(job: Job, cambios: dict[str, Any]) -> Job:
        datos = job.model_dump()
        datos.update(cambios)
        datos["actualizado_en"] = _ahora()

        nuevo_status = EstadoJob(datos["status"])
        if nuevo_status == EstadoJob.RUNNING and not datos.get("iniciado_en"):
            datos["iniciado_en"] = datos["actualizado_en"]
        if nuevo_status in ESTADOS_FINALES:
            datos["finalizado_en"] = datos["actualizado_en"]
            if datos.get("iniciado_en"):
                inicio = datetime.fromisoformat(datos["iniciado_en"])
                fin = datetime.fromisoformat(datos["finalizado_en"])
                datos["duracion_segundos"] = round((fin - inicio).total_seconds(), 3)
        return Job.model_validate(datos)
