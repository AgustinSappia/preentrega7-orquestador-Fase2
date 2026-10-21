"""
tests/conftest.py
-----------------
"Fixtures" compartidas por todos los tests. Una fixture es una función
que pytest ejecuta antes de un test para prepararle lo que necesita:
cualquier test que tenga un parámetro llamado `jobs` recibe
automáticamente un JobStore nuevo, conectado a un Redis FALSO.

`fakeredis` implementa los mismos comandos que Redis (GET, SET, WATCH,
LPUSH, ...) pero en memoria: los tests corren sin levantar ningún
servidor, y cada test arranca con un Redis vacío.
"""
import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.jobs import JobStore
from app.state import TIPOS_SERIALIZABLES


@pytest.fixture
async def redis_falso():
    cliente = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield cliente
    await cliente.aclose()


@pytest.fixture
def jobs(redis_falso) -> JobStore:
    return JobStore(redis_falso)


@pytest.fixture
def checkpointer() -> InMemorySaver:
    """Checkpointer en memoria con la MISMA lista blanca de tipos que usa
    el de Redis en producción."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_SERIALIZABLES))
