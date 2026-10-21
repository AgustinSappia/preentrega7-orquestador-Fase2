"""
tests/test_state.py
---------------------
Tests de los modelos Pydantic que definen los "artefactos" validados
que viajan por el estado compartido -- la parte de "Gestión de Estado y
Validación" de la consigna.
"""
import pytest
from pydantic import ValidationError

from app.state import AnalisisPorNoticia, NoticiaIA, ResultadoAnalisis, ResultadoInvestigacion


def test_noticia_ia_valida():
    noticia = NoticiaIA(titular="Título", fuente="Fuente", fecha="2026-01-01", resumen="Un resumen suficientemente largo.")
    assert noticia.titular == "Título"


def test_noticia_ia_rechaza_resumen_demasiado_corto():
    with pytest.raises(ValidationError):
        NoticiaIA(titular="Título", fuente="Fuente", fecha="2026-01-01", resumen="corto")


def test_resultado_investigacion_encontro_resultados():
    vacio = ResultadoInvestigacion(tema_buscado="x", noticias=[])
    assert vacio.encontro_resultados is False

    con_datos = ResultadoInvestigacion(
        tema_buscado="x",
        noticias=[
            {"titular": "T", "fuente": "F", "fecha": "2026-01-01", "resumen": "Un resumen largo."}
        ],
    )
    assert con_datos.encontro_resultados is True


def test_analisis_por_noticia_rechaza_score_fuera_de_rango():
    """El score tiene que quedar acotado entre 0 y 1 -- esto es
    justamente lo que ejercita el `ejercicio opcional` del pipeline de
    investigación/revisión: un score mal calculado tiene que fallar la
    validación en vez de colarse en el estado."""
    with pytest.raises(ValidationError):
        AnalisisPorNoticia(titular="T", sentimiento="positivo", score=1.4)


def test_resultado_analisis_sentimiento_general_literal():
    with pytest.raises(ValidationError):
        ResultadoAnalisis(sentimiento_general="muy_positivo", score_promedio=0.5, detalle=[], suficiente=True)
