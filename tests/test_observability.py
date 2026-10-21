"""
tests/test_observability.py
---------------------------
Tests de la validación de LangSmith al arrancar la API (app/observability.py).

Origen: en la primera corrida real, el .env tenía el texto de ejemplo
`LANGSMITH_API_KEY=lsv2_...` y /health igual decía "activo": true, mientras
LangSmith rechazaba todas las trazas en silencio. Estos tests cubren ese caso
y el de una key con buen formato que LangSmith rechaza. La conexión se
simula con clientes falsos: no hace falta red ni una key real.
"""
import langsmith
import pytest
from langsmith.utils import LangSmithAuthError

from app import observability

KEY_CON_FORMATO_VALIDO = "lsv2_pt_" + "a" * 40


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch: pytest.MonkeyPatch):
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    yield
    # _desactivar_tracing() apaga el tracing para todo el proceso: se restaura.
    langsmith.configure(enabled=None)


class ClienteQueAcepta:
    def list_projects(self, limit=1):
        return iter([{"name": "preentrega7-orquestador"}])


class ClienteQueRechaza:
    def list_projects(self, limit=1):
        raise LangSmithAuthError("401 Unauthorized: invalid API key")


class ClienteSinRed:
    def list_projects(self, limit=1):
        raise ConnectionError("no hay conexión")


# ---------------------------------------------------------------------------
# Validación de la configuración (sin red)
# ---------------------------------------------------------------------------
def test_tracing_apagado(monkeypatch):
    estado = observability.inicializar_observabilidad()
    assert estado["activo"] is False


def test_falta_la_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    estado = observability.inicializar_observabilidad()
    assert estado["activo"] is False
    assert "falta" in estado["motivo"]


def test_key_con_el_valor_de_ejemplo_no_cuenta_como_activa(monkeypatch):
    """El caso real: se copió .env.example sin completar la key."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_...")
    estado = observability.inicializar_observabilidad()
    assert estado["activo"] is False
    assert "ejemplo" in estado["motivo"]
    import os

    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_key_con_formato_valido_queda_pendiente_de_verificar(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", KEY_CON_FORMATO_VALIDO)
    estado = observability.inicializar_observabilidad()
    assert estado["activo"] is True
    assert estado["conexion"] == "sin verificar"
    assert estado["proyecto"] == observability.PROYECTO_POR_DEFECTO


# ---------------------------------------------------------------------------
# Verificación contra LangSmith (con clientes falsos)
# ---------------------------------------------------------------------------
async def test_langsmith_acepta_la_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", KEY_CON_FORMATO_VALIDO)
    estado = await observability.verificar_conexion(
        observability.inicializar_observabilidad(), crear_cliente=ClienteQueAcepta
    )
    assert estado["activo"] is True
    assert estado["conexion"] == "ok"


async def test_langsmith_rechaza_la_key_y_se_apaga_el_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", KEY_CON_FORMATO_VALIDO)
    estado = await observability.verificar_conexion(
        observability.inicializar_observabilidad(), crear_cliente=ClienteQueRechaza
    )
    assert estado["activo"] is False
    assert estado["conexion"] == "rechazada"
    import os

    assert os.environ["LANGSMITH_TRACING"] == "false"


async def test_sin_red_queda_activo_pero_avisa(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", KEY_CON_FORMATO_VALIDO)
    estado = await observability.verificar_conexion(
        observability.inicializar_observabilidad(), crear_cliente=ClienteSinRed
    )
    assert estado["activo"] is True
    assert estado["conexion"].startswith("sin verificar")


async def test_si_no_esta_activo_no_se_conecta():
    estado = await observability.verificar_conexion({"activo": False}, crear_cliente=ClienteQueRechaza)
    assert estado == {"activo": False}
