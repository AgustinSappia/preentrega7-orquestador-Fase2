"""
tests/test_graph.py
-------------------
Tests del grafo completo (Supervisor + especialistas + Human-in-the-loop +
Publicador), sin API keys ni Redis: se inyecta un LLM falso, una función
de decisión determinista y un `InMemorySaver` como checkpointer.

`InMemorySaver` serializa el estado exactamente igual que el
`AsyncRedisSaver` de producción (mismo serializador, misma lista blanca de
tipos), así que estos tests también validan que el estado sobreviva un
ciclo "guardar checkpoint -> reanudar".
"""
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from app.graph import construir_grafo, estado_inicial
from app.hitl import evaluar_criticidad
from app.state import TIPOS_SERIALIZABLES, ResultadoAnalisis
from tests.fakes import ModeloFalso, decisor_de_referencia, respuestas_investigar_y_analizar


def _checkpointer():
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_SERIALIZABLES))


class PublicadorFalso:
    """Registra lo que se "publicó", en vez de escribir en Redis."""

    def __init__(self):
        self.publicados = []

    async def __call__(self, reporte):
        self.publicados.append(reporte)


def _config(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}}


# ---------------------------------------------------------------------------
# Regla de criticidad (función pura)
# ---------------------------------------------------------------------------
def _analisis(sentimiento: str) -> ResultadoAnalisis:
    return ResultadoAnalisis(sentimiento_general=sentimiento, score_promedio=0.5, detalle=[], suficiente=True)


def test_criticidad_sentimiento_negativo_es_critico():
    critica, motivo = evaluar_criticidad({"analisis": _analisis("negativo"), "forzar_aprobacion": False})
    assert critica is True
    assert "NEGATIVO" in motivo


def test_criticidad_sentimiento_positivo_no_es_critico():
    critica, _ = evaluar_criticidad({"analisis": _analisis("positivo"), "forzar_aprobacion": False})
    assert critica is False


def test_criticidad_forzada_por_el_cliente():
    critica, _ = evaluar_criticidad({"analisis": _analisis("positivo"), "forzar_aprobacion": True})
    assert critica is True


# ---------------------------------------------------------------------------
# Escenario 1: no crítico -> se publica sin pausa
# ---------------------------------------------------------------------------
async def test_tarea_no_critica_se_publica_sin_pausa():
    publicador = PublicadorFalso()
    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("modelos")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=_checkpointer(),
        publicar=publicador,
    )
    resultado = await grafo.ainvoke(estado_inicial("modelos"), _config("job-feliz"))

    assert "__interrupt__" not in resultado
    assert resultado["analisis"].sentimiento_general == "positivo"
    assert resultado["aprobacion"].requerida is False
    assert resultado["aprobacion"].aprobado is True
    assert resultado["reporte"] is not None
    assert len(publicador.publicados) == 1


# ---------------------------------------------------------------------------
# Escenario 2: crítico -> pausa -> aprobado -> se publica
# ---------------------------------------------------------------------------
async def test_tarea_critica_pausa_y_se_publica_al_aprobar():
    publicador = PublicadorFalso()
    checkpointer = _checkpointer()
    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("regulación")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
        publicar=publicador,
    )
    config = _config("job-critico")

    # 1) Primera invocación: tiene que DETENERSE en AprobacionHumana.
    resultado = await grafo.ainvoke(estado_inicial("regulación"), config)
    assert "__interrupt__" in resultado
    payload = resultado["__interrupt__"][0].value
    assert payload["tipo"] == "aprobacion_publicacion"
    assert payload["propuesta"]["sentimiento_general"] == "negativo"
    assert publicador.publicados == []  # todavía NO se publicó nada

    snapshot = await grafo.aget_state(config)
    assert snapshot.next == ("AprobacionHumana",)

    # 2) Se reanuda con la decisión humana (como hace POST /approve).
    resultado = await grafo.ainvoke(
        Command(resume={"aprobado": True, "comentario": "OK para publicar", "revisor": "rayo"}), config
    )
    assert "__interrupt__" not in resultado
    assert resultado["aprobacion"].requerida is True
    assert resultado["aprobacion"].aprobado is True
    assert resultado["aprobacion"].revisor == "rayo"
    assert len(publicador.publicados) == 1
    assert publicador.publicados[0].sentimiento_general == "negativo"


async def test_reanudar_con_un_grafo_nuevo_mismo_checkpointer():
    """Simula que la API se reinició entre la pausa y la aprobación: el
    grafo que reanuda es OTRA instancia, y todo lo que sabe sale del
    checkpointer (en producción, de Redis)."""
    checkpointer = _checkpointer()
    grafo_antes = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("regulación")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
        publicar=PublicadorFalso(),
    )
    config = _config("job-reinicio")
    await grafo_antes.ainvoke(estado_inicial("regulación"), config)

    publicador = PublicadorFalso()
    grafo_despues = construir_grafo(
        modelo=ModeloFalso(responses=[]),  # no debería hacer falta ningún LLM
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=checkpointer,
        publicar=publicador,
    )
    resultado = await grafo_despues.ainvoke(Command(resume={"aprobado": True}), config)
    assert resultado["investigacion"].tema_buscado == "regulación"
    assert len(publicador.publicados) == 1


# ---------------------------------------------------------------------------
# Escenario 3: crítico -> pausa -> rechazado -> NO se publica
# ---------------------------------------------------------------------------
async def test_tarea_critica_rechazada_no_se_publica():
    publicador = PublicadorFalso()
    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("regulación")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=_checkpointer(),
        publicar=publicador,
    )
    config = _config("job-rechazado")
    await grafo.ainvoke(estado_inicial("regulación"), config)
    resultado = await grafo.ainvoke(Command(resume={"aprobado": False, "comentario": "No publicar"}), config)

    assert resultado["aprobacion"].aprobado is False
    assert resultado["reporte"] is None
    assert publicador.publicados == []


async def test_forzar_aprobacion_pausa_aunque_sea_positivo():
    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("modelos")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=_checkpointer(),
        publicar=PublicadorFalso(),
    )
    resultado = await grafo.ainvoke(estado_inicial("modelos", forzar_aprobacion=True), _config("job-forzado"))
    assert "__interrupt__" in resultado


async def test_decision_humana_invalida_falla_validacion():
    """El nodo valida con Pydantic lo que llega al reanudar."""
    grafo = construir_grafo(
        modelo=ModeloFalso(responses=respuestas_investigar_y_analizar("regulación")),
        decidir_siguiente_paso=decisor_de_referencia,
        checkpointer=_checkpointer(),
        publicar=PublicadorFalso(),
    )
    config = _config("job-invalido")
    await grafo.ainvoke(estado_inicial("regulación"), config)
    with pytest.raises(Exception):
        await grafo.ainvoke(Command(resume={"comentario": "falta el campo aprobado"}), config)


# ---------------------------------------------------------------------------
# Escenarios heredados de la Pre-entrega 6
# ---------------------------------------------------------------------------
async def test_ciclo_de_refinamiento_termina_publicando():
    modelo = ModeloFalso(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "buscar_noticias_ia", "args": {"tema": "blockchain cuántico", "maximo": 5}, "id": "c1"}
                ],
            ),
            AIMessage(content="No encontré noticias."),
            # (el Analista hace short-circuit con 0 noticias: no consume respuestas)
            *respuestas_investigar_y_analizar("ia"),
        ]
    )
    publicador = PublicadorFalso()
    grafo = construir_grafo(
        modelo=modelo, decidir_siguiente_paso=decisor_de_referencia, checkpointer=_checkpointer(), publicar=publicador
    )
    resultado = await grafo.ainvoke(estado_inicial("blockchain cuántico"), _config("job-refinamiento"))

    assert resultado["investigacion"].tema_buscado == "ia"
    assert resultado["analisis"].suficiente is True
    # "ia" agrupa las 5 noticias -> sentimiento neutral -> no crítico.
    assert len(publicador.publicados) == 1


async def test_corte_por_contador_no_pasa_por_aprobacion():
    async def decisor_terco(state):
        return "Investigador"

    modelo = ModeloFalso(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": "x"}, "id": "c1"}]),
            AIMessage(content="Nada."),
        ]
    )
    publicador = PublicadorFalso()
    grafo = construir_grafo(
        modelo=modelo,
        decidir_siguiente_paso=decisor_terco,
        max_pasos=3,
        checkpointer=_checkpointer(),
        publicar=publicador,
    )
    resultado = await grafo.ainvoke(estado_inicial("tema inexistente"), _config("job-terco"))

    assert resultado["pasos"] == 4
    assert resultado["task_completed"] is True
    assert resultado["aprobacion"] is None
    assert publicador.publicados == []
