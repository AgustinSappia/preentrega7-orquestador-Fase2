"""
tests/test_checkpointer_redis.py
--------------------------------
Test de regresión del error que apareció en la primera corrida real con
Redis 8:

    AttributeError: 'dict' object has no attribute 'sentimiento_general'

¿Qué pasaba? El checkpointer de Redis guarda los checkpoints como JSON.
Al reanudar un job después de la aprobación humana, los artefactos
Pydantic (ResultadoAnalisis, etc.) volvían como `dict`, porque el
serializador no tenía permitidas esas clases en modo JSON. Los demás tests
no lo detectaban: usan `InMemorySaver`, que serializa con msgpack.

Este test usa el código REAL de `AsyncRedisSaver` que convierte el estado a
JSON y lo vuelve a armar (`_dump_checkpoint` y
`_deserialize_channel_values`), con el serializador de producción
(`serializador_redis()`). No necesita un Redis levantado: el saver no se
conecta hasta que se usa.
"""
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.graph import construir_grafo, estado_inicial, serializador_redis
from app.hitl import evaluar_criticidad
from app.state import NoticiaIA, ResultadoAnalisis, ResultadoInvestigacion
from tests.fakes import ModeloFalso, decisor_de_referencia, respuestas_investigar_y_analizar


def _ida_y_vuelta_por_redis(valores: dict) -> dict:
    """Simula guardar el estado en Redis y leerlo al reanudar."""
    saver = AsyncRedisSaver(redis_url="redis://localhost:6379")
    saver.serde = serializador_redis()
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = valores
    guardado = saver._dump_checkpoint(checkpoint)
    return saver._deserialize_channel_values(guardado["channel_values"])


async def _estado_pausado_en_aprobacion(checkpointer) -> dict:
    """El estado REAL del grafo justo cuando se pausa en AprobacionHumana."""
    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("regulación")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "job-redis"}}
    await grafo.ainvoke(estado_inicial("regulación"), config)
    return (await grafo.aget_state(config)).values


async def test_los_artefactos_vuelven_como_objetos_pydantic(checkpointer):
    original = await _estado_pausado_en_aprobacion(checkpointer)
    leido = _ida_y_vuelta_por_redis(original)

    assert isinstance(leido["analisis"], ResultadoAnalisis)
    assert isinstance(leido["investigacion"], ResultadoInvestigacion)
    assert all(isinstance(n, NoticiaIA) for n in leido["investigacion"].noticias)
    assert leido["analisis"] == original["analisis"]
    assert leido["investigacion"] == original["investigacion"]


async def test_el_nodo_de_aprobacion_puede_usar_el_estado_leido_de_redis(checkpointer):
    """Lo que fallaba en la corrida real: al reanudar, el nodo accede a
    `analisis.sentimiento_general`."""
    leido = _ida_y_vuelta_por_redis(await _estado_pausado_en_aprobacion(checkpointer))
    es_critica, motivo = evaluar_criticidad(leido)
    assert es_critica is True
    assert "NEGATIVO" in motivo


async def test_aprobacion_y_reporte_tambien_sobreviven(checkpointer):
    """El estado final (después de aprobar) incluye RegistroAprobacion y
    ReportePublicado: también tienen que volver como objetos."""
    from langgraph.types import Command

    from app.state import RegistroAprobacion, ReportePublicado

    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("regulación")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "job-redis-final"}}
    await grafo.ainvoke(estado_inicial("regulación"), config)
    await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "rayo"}), config)

    leido = _ida_y_vuelta_por_redis((await grafo.aget_state(config)).values)
    assert isinstance(leido["aprobacion"], RegistroAprobacion)
    assert leido["aprobacion"].revisor == "rayo"
    assert isinstance(leido["reporte"], ReportePublicado)
