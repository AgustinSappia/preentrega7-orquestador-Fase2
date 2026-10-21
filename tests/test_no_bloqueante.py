"""
tests/test_no_bloqueante.py
---------------------------
Prueba con un servidor HTTP REAL (uvicorn) que la API no bloquea el event
loop -- el criterio de mayor peso de la rúbrica (30%).

Se usa un grafo falso que tarda 1 segundo por tarea (simula la espera al
LLM con `await asyncio.sleep`). Se lanzan 5 POST /tasks a la vez y se
verifica que:

1. Los 5 POST responden enseguida (bastante menos que 1 s cada uno): la
   API devuelve el job_id sin esperar a que el grafo termine.
2. Mientras las 5 tareas corren, GET /health sigue respondiendo rápido.
3. Las 5 tareas terminan en ~1 s en total, no en ~5 s: corren de forma
   CONCURRENTE dentro del mismo event loop.
"""
import asyncio
import socket
import time

import httpx
import uvicorn

from app.jobs import JobStore
from app.main import crear_app

DURACION_TAREA = 1.0


class GrafoLento:
    """Simula un grafo cuyo LLM tarda DURACION_TAREA segundos en responder."""

    async def ainvoke(self, entrada, config=None):
        await asyncio.sleep(DURACION_TAREA)  # espera ASÍNCRONA: libera el event loop
        return {"topic": entrada["topic"], "pasos": 3, "investigacion": None, "analisis": None, "aprobacion": None, "reporte": None}


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_cinco_peticiones_concurrentes_no_bloquean_la_api(jobs: JobStore):
    app = crear_app(lifespan_fn=None)
    app.state.jobs = jobs
    app.state.grafo = GrafoLento()
    app.state.observabilidad = None

    puerto = _puerto_libre()
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=puerto, log_level="warning", lifespan="off"))
    tarea_servidor = asyncio.create_task(servidor.serve())
    while not servidor.started:
        await asyncio.sleep(0.01)

    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{puerto}") as cliente:
            inicio = time.perf_counter()

            async def post(tema: str):
                t0 = time.perf_counter()
                r = await cliente.post("/tasks", json={"topic": tema})
                return r, time.perf_counter() - t0

            respuestas = await asyncio.gather(*[post(t) for t in ["modelos", "regulación", "robótica", "adopción", "agentes"]])

            # 1) Los POST responden sin esperar a la tarea.
            for r, latencia in respuestas:
                assert r.status_code == 202
                assert latencia < DURACION_TAREA / 2, f"POST tardó {latencia:.2f}s: ¡la API esperó a la tarea!"

            # 2) La API sigue atendiendo mientras las tareas corren.
            t0 = time.perf_counter()
            health = await cliente.get("/health")
            assert health.status_code == 200
            assert time.perf_counter() - t0 < 0.3

            ids = [r.json()["job_id"] for r, _ in respuestas]
            estados = [(await cliente.get(f"/tasks/{i}")).json()["status"] for i in ids]
            assert all(e in {"PENDING", "RUNNING"} for e in estados), estados

            # 3) Esperar a que terminen las 5 (polling, como haría un cliente).
            while True:
                estados = [(await cliente.get(f"/tasks/{i}")).json()["status"] for i in ids]
                if all(e == "DONE" for e in estados):
                    break
                assert time.perf_counter() - inicio < 5 * DURACION_TAREA, "las tareas no corren en paralelo"
                await asyncio.sleep(0.05)

            total = time.perf_counter() - inicio
            # En serie serían ~5 s; en paralelo, poco más de 1 s.
            assert total < 2.5 * DURACION_TAREA, f"total {total:.2f}s"
    finally:
        servidor.should_exit = True
        await tarea_servidor
