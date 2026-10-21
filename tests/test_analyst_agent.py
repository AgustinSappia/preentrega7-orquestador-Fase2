"""
tests/test_analyst_agent.py
-----------------------------
Tests del Agente Analista: la herramienta `analizar_sentimiento_noticias`
en forma aislada, y la función `analizar()` end-to-end -- incluyendo el
short-circuit cuando no hay noticias para analizar.
"""
from langchain_core.messages import AIMessage

from app.agents.analyst_agent import analizar, analizar_sentimiento_noticias
from app.state import ResultadoAnalisis, ResultadoInvestigacion
from tests.fakes import ModeloFalso

_NOTICIA_POSITIVA = {
    "titular": "Buenas noticias",
    "fuente": "Fuente",
    "fecha": "2026-01-01",
    "resumen": "Gran mejora y entusiasmo, resultados récord con mucho optimismo.",
}
_NOTICIA_NEGATIVA = {
    "titular": "Malas noticias",
    "fuente": "Fuente",
    "fecha": "2026-01-01",
    "resumen": "Preocupación por el riesgo y fallas que generan debate y crítica.",
}


# ---------------------------------------------------------------------------
# La herramienta en sí, sin ningún LLM de por medio.
# ---------------------------------------------------------------------------
def test_analizar_sentimiento_lista_vacia_es_sin_datos():
    resultado = analizar_sentimiento_noticias.invoke({"noticias": []})
    assert resultado["sentimiento_general"] == "sin_datos"
    assert resultado["detalle"] == []


def test_analizar_sentimiento_detecta_positivo_y_negativo():
    resultado = analizar_sentimiento_noticias.invoke(
        {"noticias": [_NOTICIA_POSITIVA, _NOTICIA_NEGATIVA]}
    )
    detalle_por_titular = {item["titular"]: item for item in resultado["detalle"]}
    assert detalle_por_titular["Buenas noticias"]["sentimiento"] == "positivo"
    assert detalle_por_titular["Malas noticias"]["sentimiento"] == "negativo"


def test_analizar_sentimiento_scores_quedan_acotados_entre_0_y_1():
    resultado = analizar_sentimiento_noticias.invoke({"noticias": [_NOTICIA_POSITIVA]})
    for item in resultado["detalle"]:
        assert 0.0 <= item["score"] <= 1.0
    assert 0.0 <= resultado["score_promedio"] <= 1.0


# ---------------------------------------------------------------------------
# `analizar()`: el punto de entrada que usa el nodo del grafo.
# ---------------------------------------------------------------------------
async def test_analizar_con_investigacion_vacia_hace_short_circuit():
    """Si no hay noticias, `analizar()` ni siquiera tiene que llamar al
    modelo -- si lo hiciera, este test fallaría porque el modelo falso
    no tiene ninguna respuesta scripteada."""
    modelo = ModeloFalso(responses=[])
    investigacion = ResultadoInvestigacion(tema_buscado="x", noticias=[])
    resultado = await analizar(modelo, investigacion)
    assert isinstance(resultado, ResultadoAnalisis)
    assert resultado.sentimiento_general == "sin_datos"
    assert resultado.suficiente is False


async def test_analizar_con_noticias_devuelve_resultado_validado():
    noticias_payload = [_NOTICIA_POSITIVA]
    modelo = ModeloFalso(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analizar_sentimiento_noticias",
                        "args": {"noticias": noticias_payload},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="Listo."),
        ]
    )
    investigacion = ResultadoInvestigacion(tema_buscado="x", noticias=noticias_payload)
    resultado = await analizar(modelo, investigacion)
    assert resultado.sentimiento_general == "positivo"
    assert resultado.suficiente is True
    assert len(resultado.detalle) == 1
