"""
agents/analyst_agent.py
-------------------------
Agente especialista "Analista": recibe las noticias que encontró el
Investigador y evalúa su sentimiento (positivo / neutral / negativo),
usando una heurística léxica simulada (sin ningún servicio externo de
NLP) dentro de su propia herramienta.
"""

import json
from typing import List

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from pydantic import ValidationError

from app.state import ResultadoAnalisis, ResultadoInvestigacion

# ---------------------------------------------------------------------------
# Heurística de sentimiento simulada (léxico de palabras clave)
# ---------------------------------------------------------------------------
_PALABRAS_POSITIVAS = [
    "mejora", "mejoras", "positiv", "optimismo", "crece", "creci",
    "récord", "satisfacción", "entusiasmo", "oportunidad", "beneficio",
    "ahorro",
]
_PALABRAS_NEGATIVAS = [
    "preocup", "alucinaci", "sesgo", "riesgo", "fall", "incorrect",
    "advierten", "crítica", "amenaza", "problema", "debate",
]


def _clasificar_sentimiento(texto: str) -> tuple:
    """Heurística simple: cuenta palabras positivas y negativas en el
    texto y las convierte en un score entre 0 y 1. No es un modelo de
    NLP real -- es intencionalmente simple, para que el resultado sea
    100% determinístico y fácil de testear sin ninguna API."""
    texto_lower = texto.lower()
    positivas = sum(1 for palabra in _PALABRAS_POSITIVAS if palabra in texto_lower)
    negativas = sum(1 for palabra in _PALABRAS_NEGATIVAS if palabra in texto_lower)
    diferencia = positivas - negativas

    score = max(0.0, min(1.0, 0.5 + 0.15 * diferencia))

    if score > 0.6:
        sentimiento = "positivo"
    elif score < 0.4:
        sentimiento = "negativo"
    else:
        sentimiento = "neutral"

    return sentimiento, score


@tool
def analizar_sentimiento_noticias(noticias: List[dict]) -> dict:
    """Analiza el sentimiento de una lista de noticias y devuelve un resumen agregado.

    Usá esta herramienta cuando ya tengas una lista de noticias (cada una
    con al menos "titular" y "resumen") y necesites evaluar si el tono
    general es positivo, neutral o negativo. Pasale la lista de noticias
    exactamente como te la dieron, sin resumirla ni modificarla.

    Args:
        noticias: Lista de diccionarios, cada uno con las claves
            "titular", "fuente", "fecha" y "resumen".

    Returns:
        Un diccionario con "detalle" (sentimiento y score de 0 a 1 por
        cada noticia individual) y los agregados "sentimiento_general" y
        "score_promedio". Si la lista está vacía, devuelve
        "sentimiento_general": "sin_datos".
    """
    if not noticias:
        return {"detalle": [], "sentimiento_general": "sin_datos", "score_promedio": 0.0}

    detalle = []
    for noticia in noticias:
        sentimiento, score = _clasificar_sentimiento(noticia.get("resumen", ""))
        detalle.append(
            {"titular": noticia.get("titular", ""), "sentimiento": sentimiento, "score": score}
        )

    score_promedio = sum(item["score"] for item in detalle) / len(detalle)
    if score_promedio > 0.6:
        general = "positivo"
    elif score_promedio < 0.4:
        general = "negativo"
    else:
        general = "neutral"

    return {
        "detalle": detalle,
        "sentimiento_general": general,
        "score_promedio": round(score_promedio, 3),
    }


TOOLS = [analizar_sentimiento_noticias]

PROMPT_ANALISTA = (
    "Sos un agente especializado ÚNICAMENTE en analizar el sentimiento de "
    "noticias que ya fueron investigadas por otro agente. Tu única tarea "
    "es llamar a la herramienta analizar_sentimiento_noticias con la "
    "lista de noticias EXACTA que te dan en el mensaje, sin modificarla. "
    "No investigues noticias nuevas -- esa tarea es de otro agente "
    "especialista. No inventes un sentimiento sin haber llamado a la "
    "herramienta."
)


def construir_agente_analista(modelo):
    """Crea el agente ReAct del Analista, con una sola herramienta
    disponible: no puede, por ejemplo, intentar buscar noticias nuevas."""
    return create_react_agent(modelo, tools=TOOLS, prompt=PROMPT_ANALISTA)


# [Pre-entrega 7] Decorador de observabilidad: cuando LangSmith está
# activo, cada llamada a esta función aparece como un "span" propio
# dentro de la traza del job (con su entrada, salida, duración y los
# tokens de las llamadas al LLM que haga adentro). Si LangSmith no
# está configurado, el decorador no hace nada y no rompe nada.
@traceable(name="agente_analista", run_type="chain", tags=["especialista", "Analista"])
async def analizar(modelo, investigacion: ResultadoInvestigacion) -> ResultadoAnalisis:
    """Punto de entrada que usa el nodo del grafo principal.

    Recibe el ResultadoInvestigacion COMPLETO (no el historial de
    mensajes del orquestador) y solo le pasa al agente lo que
    necesita -- la lista de noticias -- evitando la "Contaminación de
    Contexto" que señala la consigna.
    """
    if not investigacion.encontro_resultados:
        # No hay nada que analizar: ni siquiera hace falta invocar al
        # LLM. `suficiente=False` es la señal que el Supervisor usa para
        # decidir si vale la pena reintentar la investigación.
        return ResultadoAnalisis(
            sentimiento_general="sin_datos", score_promedio=0.0, detalle=[], suficiente=False
        )

    agente = construir_agente_analista(modelo)
    noticias_payload = [noticia.model_dump() for noticia in investigacion.noticias]

    entrada = {
        "messages": [
            HumanMessage(
                content=(
                    "Analizá el sentimiento de estas noticias usando la "
                    "herramienta analizar_sentimiento_noticias. Pasale "
                    "exactamente esta lista como argumento 'noticias':\n"
                    f"{json.dumps(noticias_payload, ensure_ascii=False)}"
                )
            )
        ]
    }
    resultado = await agente.ainvoke(entrada)

    mensajes_herramienta = [m for m in resultado["messages"] if isinstance(m, ToolMessage)]
    if not mensajes_herramienta:
        return ResultadoAnalisis(
            sentimiento_general="sin_datos", score_promedio=0.0, detalle=[], suficiente=False
        )

    payload = json.loads(mensajes_herramienta[-1].content)

    try:
        return ResultadoAnalisis(
            sentimiento_general=payload["sentimiento_general"],
            score_promedio=payload["score_promedio"],
            detalle=payload["detalle"],
            # "Suficiente" refleja el RESULTADO real del análisis, no
            # solo que la herramienta se haya ejecutado: si terminó en
            # "sin_datos" (por ejemplo porque la lista que recibió venía
            # vacía), no es un análisis suficiente para que el Supervisor
            # pueda cerrar la conversación con eso.
            suficiente=payload["sentimiento_general"] != "sin_datos",
        )
    except (ValidationError, KeyError):
        return ResultadoAnalisis(
            sentimiento_general="sin_datos", score_promedio=0.0, detalle=[], suficiente=False
        )
