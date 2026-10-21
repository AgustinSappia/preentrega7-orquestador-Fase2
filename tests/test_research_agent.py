"""
tests/test_research_agent.py
------------------------------
Tests del Agente Investigador: la herramienta `buscar_noticias_ia` en
forma aislada, y la función `investigar()` end-to-end con un LLM falso.
"""
from langchain_core.messages import AIMessage

from app.agents.research_agent import buscar_noticias_ia, investigar
from app.state import ResultadoInvestigacion
from tests.fakes import ModeloFalso


# ---------------------------------------------------------------------------
# La herramienta en sí, sin ningún LLM de por medio.
# ---------------------------------------------------------------------------
def test_buscar_noticias_encuentra_por_etiqueta():
    resultado = buscar_noticias_ia.invoke({"tema": "robótica"})
    assert resultado["tema_buscado"] == "robótica"
    assert len(resultado["noticias"]) == 1
    assert "robótica" in resultado["noticias"][0]["titular"].lower()


def test_buscar_noticias_no_distingue_mayusculas():
    resultado = buscar_noticias_ia.invoke({"tema": "ROBÓTICA"})
    assert len(resultado["noticias"]) == 1


def test_buscar_noticias_sin_resultados_devuelve_lista_vacia():
    """Si el tema no matchea nada, la herramienta tiene que decirlo
    honestamente (lista vacía) en vez de inventar una noticia."""
    resultado = buscar_noticias_ia.invoke({"tema": "blockchain cuántico"})
    assert resultado["noticias"] == []


def test_buscar_noticias_respeta_el_maximo():
    resultado = buscar_noticias_ia.invoke({"tema": "ia", "maximo": 2})
    assert len(resultado["noticias"]) == 2


# ---------------------------------------------------------------------------
# `investigar()`: el punto de entrada que usa el nodo del grafo.
# ---------------------------------------------------------------------------
async def test_investigar_devuelve_resultado_validado():
    modelo = ModeloFalso(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": "modelos", "maximo": 5}, "id": "c1"}],
            ),
            AIMessage(content="Encontré noticias sobre modelos."),
        ]
    )
    resultado = await investigar(modelo, "modelos")
    assert isinstance(resultado, ResultadoInvestigacion)
    assert resultado.tema_buscado == "modelos"
    assert resultado.encontro_resultados is True
    assert len(resultado.noticias) == 2


async def test_investigar_sin_resultados_no_rompe():
    # "blockchain cuántico" es deliberadamente un término que no matchea
    # ninguna de las noticias simuladas (a diferencia de un término de
    # una sola letra como "x", que por ser búsqueda por substring puede
    # matchear palabras sueltas sin querer).
    modelo = ModeloFalso(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "buscar_noticias_ia", "args": {"tema": "blockchain cuántico", "maximo": 5}, "id": "c1"}
                ],
            ),
            AIMessage(content="No encontré nada."),
        ]
    )
    resultado = await investigar(modelo, "blockchain cuántico")
    assert resultado.encontro_resultados is False
    assert resultado.noticias == []


async def test_investigar_si_el_agente_nunca_llama_la_herramienta():
    """Si por algún motivo el LLM responde directamente sin usar la
    herramienta, `investigar()` tiene que degradar con gracia en vez de
    romper -- lo tratamos como 'no se encontró nada'."""
    modelo = ModeloFalso(responses=[AIMessage(content="No hace falta buscar nada.")])
    resultado = await investigar(modelo, "modelos")
    assert resultado.noticias == []
    assert resultado.tema_buscado == "modelos"


def test_buscar_noticias_acepta_frases_ademas_de_palabras():
    """[Pre-entrega 7] Un LLM real a veces pasa una frase en vez de una
    palabra suelta: igual tiene que encontrar la noticia."""
    resultado = buscar_noticias_ia.invoke({"tema": "noticias sobre regulación de la IA"})
    assert len(resultado["noticias"]) == 1
    assert "Unión Europea" in resultado["noticias"][0]["titular"]


def test_buscar_noticias_frase_sin_coincidencias_sigue_vacia():
    resultado = buscar_noticias_ia.invoke({"tema": "computación cuántica blockchain"})
    assert resultado["noticias"] == []
