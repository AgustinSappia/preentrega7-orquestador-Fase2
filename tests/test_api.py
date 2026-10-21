"""
tests/test_api.py
-----------------
Tests de los endpoints HTTP, sin levantar ningún servidor ni Redis real:
`httpx.ASGITransport` le "habla" directamente a la app de FastAPI en
memoria, y en `app.state` se cargan un JobStore sobre fakeredis y un grafo
con LLM falso.

Detalle útil: con ASGITransport, las BackgroundTasks terminan ANTES de que
el cliente reciba la respuesta. Por eso, en estos tests, un GET justo
después del POST ya ve el estado final. (Que la API de verdad NO espera a
la tarea lo prueba aparte tests/test_no_bloqueante.py, con un servidor real.)
"""
import httpx
import pytest

from app.graph import construir_grafo
from app.jobs import JobStore
from app.main import crear_app
from tests.fakes import ModeloFalso, decisor_de_referencia, respuestas_investigar_y_analizar


class GrafoQueExplota:
    async def ainvoke(self, entrada, config=None):
        raise ValueError("respuesta inválida del LLM")


def _app_con(jobs: JobStore, grafo):
    app = crear_app(lifespan_fn=None)
    app.state.jobs = jobs
    app.state.grafo = grafo
    app.state.observabilidad = {"plataforma": "langsmith", "activo": False}
    return app


def _grafo(tema: str, jobs: JobStore, checkpointer):
    return construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar(tema)),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
        publicar=jobs.publicar_reporte,
    )


@pytest.fixture
def cliente_factory():
    clientes = []

    def crear(app):
        cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clientes.append(cliente)
        return cliente

    yield crear


# ---------------------------------------------------------------------------
# POST /tasks + GET /tasks/{id}
# ---------------------------------------------------------------------------
async def test_post_devuelve_202_con_job_id(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    respuesta = await cliente.post("/tasks", json={"topic": "modelos"})

    assert respuesta.status_code == 202
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "PENDING"  # se respondió ANTES de correr el grafo
    assert cuerpo["links"]["estado"] == f"/tasks/{cuerpo['job_id']}"


async def test_tarea_no_critica_termina_done(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    job_id = (await cliente.post("/tasks", json={"topic": "modelos"})).json()["job_id"]

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["status"] == "DONE"
    assert estado["resultado"]["analisis"]["sentimiento_general"] == "positivo"
    assert estado["resultado"]["reporte_publicado"] is not None

    reportes = (await cliente.get("/reportes")).json()
    assert len(reportes) == 1


async def test_get_job_inexistente_devuelve_404(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    assert (await cliente.get("/tasks/no-existe")).status_code == 404


async def test_post_invalido_devuelve_422(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    assert (await cliente.post("/tasks", json={"topic": ""})).status_code == 422
    assert (await cliente.post("/tasks", json={})).status_code == 422


async def test_error_en_el_grafo_se_ve_como_failed(jobs, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, GrafoQueExplota()))
    job_id = (await cliente.post("/tasks", json={"topic": "modelos"})).json()["job_id"]

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["status"] == "FAILED"
    assert "ValueError" in estado["error"]


async def test_listar_tareas(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    job_id = (await cliente.post("/tasks", json={"topic": "modelos"})).json()["job_id"]
    listado = (await cliente.get("/tasks")).json()
    assert [j["job_id"] for j in listado] == [job_id]


async def test_health(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    cuerpo = (await cliente.get("/health")).json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["redis"] is True


# ---------------------------------------------------------------------------
# Human-in-the-loop por HTTP
# ---------------------------------------------------------------------------
async def test_flujo_hitl_completo_por_http(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("regulación", jobs, checkpointer)))
    job_id = (await cliente.post("/tasks", json={"topic": "regulación"})).json()["job_id"]

    # 1) Queda pausado esperando aprobación, con la propuesta a revisar.
    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["status"] == "AWAITING_APPROVAL"
    assert estado["aprobacion_pendiente"]["propuesta"]["sentimiento_general"] == "negativo"
    assert (await cliente.get("/reportes")).json() == []

    # 2) Se aprueba -> 202 -> termina DONE y publica.
    respuesta = await cliente.post(
        f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "OK", "revisor": "rayo"}
    )
    assert respuesta.status_code == 202

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["status"] == "DONE"
    assert estado["resultado"]["aprobacion"]["revisor"] == "rayo"
    assert len((await cliente.get("/reportes")).json()) == 1

    # 3) Aprobar de nuevo un job que ya terminó -> 409.
    respuesta = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert respuesta.status_code == 409


async def test_rechazo_por_http(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("regulación", jobs, checkpointer)))
    job_id = (await cliente.post("/tasks", json={"topic": "regulación"})).json()["job_id"]
    await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "No"})

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["status"] == "REJECTED"
    assert (await cliente.get("/reportes")).json() == []


async def test_aprobar_job_inexistente_devuelve_404(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    respuesta = await cliente.post("/tasks/no-existe/approve", json={"aprobado": True})
    assert respuesta.status_code == 404


async def test_aprobar_job_que_no_espera_aprobacion_devuelve_409(jobs, checkpointer, cliente_factory):
    cliente = cliente_factory(_app_con(jobs, _grafo("modelos", jobs, checkpointer)))
    job_id = (await cliente.post("/tasks", json={"topic": "modelos"})).json()["job_id"]  # termina DONE
    respuesta = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert respuesta.status_code == 409
    assert "DONE" in respuesta.json()["detail"]
