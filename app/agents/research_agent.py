"""
agents/research_agent.py
--------------------------
Agente especialista "Investigador": busca noticias de IA (simuladas,
en memoria -- sin Tavily ni ninguna API externa) relacionadas con el
tema que le pida el Supervisor.

Se construye con `create_react_agent` de LangGraph (como sugiere la
consigna) en vez de armar el ciclo ReAct a mano -- ya lo hicimos a mano
en la Pre-entrega 5, así que acá usamos el helper que hace exactamente
lo mismo (bind_tools + ToolNode + arista condicional + ciclo) con menos
código repetido.
"""

import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from pydantic import ValidationError

from app.state import ResultadoInvestigacion

# ---------------------------------------------------------------------------
# "Base de datos" simulada de noticias de IA
# ---------------------------------------------------------------------------
_NOTICIAS_IA = [
    {
        "titular": "OpenAI presenta un nuevo modelo con razonamiento mejorado",
        "fuente": "TechCrunch",
        "fecha": "2026-09-15",
        "resumen": (
            "La compañía anunció mejoras significativas en benchmarks de "
            "razonamiento matemático y código, con reacciones mayormente "
            "positivas y entusiasmo de la comunidad desarrolladora."
        ),
        "etiquetas": ["modelos", "lanzamiento", "openai", "razonamiento", "ia"],
    },
    {
        "titular": "Preocupación regulatoria crece en la Unión Europea por el uso de IA en contratación laboral",
        "fuente": "Reuters",
        "fecha": "2026-09-10",
        "resumen": (
            "Organizaciones de derechos civiles advierten sobre sesgos "
            "algorítmicos preocupantes en sistemas automatizados de "
            "selección de personal, pidiendo regulación urgente."
        ),
        "etiquetas": ["regulación", "ética", "empleo", "ia"],
    },
    {
        "titular": "Empresa de robótica anuncia ronda de inversión récord impulsada por IA generativa",
        "fuente": "Bloomberg",
        "fecha": "2026-09-18",
        "resumen": (
            "Inversionistas muestran gran optimismo sobre la integración "
            "de modelos generativos en robots físicos para tareas "
            "industriales, en una ronda histórica para el sector."
        ),
        "etiquetas": ["inversión", "robótica", "modelos", "ia"],
    },
    {
        "titular": "Investigadores publican paper sobre fallas de alucinación en agentes autónomos",
        "fuente": "MIT Technology Review",
        "fecha": "2026-09-05",
        "resumen": (
            "El estudio documenta casos preocupantes donde agentes de IA "
            "tomaron decisiones incorrectas por falta de verificación "
            "externa, generando un fuerte debate en la comunidad académica."
        ),
        "etiquetas": ["investigación", "seguridad", "agentes", "ia"],
    },
    {
        "titular": "Adopción de asistentes de IA en pymes crece un 40% este trimestre",
        "fuente": "Forbes",
        "fecha": "2026-09-20",
        "resumen": (
            "Pequeñas y medianas empresas reportan ahorros de tiempo "
            "considerables y buena satisfacción, aunque algunas señalan "
            "una curva de aprendizaje pronunciada al principio."
        ),
        "etiquetas": ["adopción", "negocios", "pymes", "ia"],
    },
]


_PALABRAS_VACIAS = {"sobre", "noticias", "noticia", "acerca", "para", "desde", "entre", "como", "ultimas", "últimas"}


def _palabras_clave(tema: str) -> list[str]:
    """[Pre-entrega 7] Separa el tema en palabras clave.

    En la Pre-entrega 6 la búsqueda pedía que el tema COMPLETO apareciera
    textual en la noticia: con un LLM real, que a veces pasa una frase
    ("regulación de la IA") en vez de una palabra ("regulación"), eso
    devolvía 0 resultados. Ahora alcanza con que aparezca alguna palabra
    significativa (4+ letras, sin palabras vacías como "sobre"). Si el
    tema no tiene ninguna (ej. "ia"), se busca el tema completo.
    """
    busqueda = tema.strip().lower()
    palabras = [p.strip(".,;:¿?¡!\"'") for p in busqueda.split()]
    significativas = [p for p in palabras if len(p) >= 4 and p not in _PALABRAS_VACIAS]
    return significativas or [busqueda]


@tool
def buscar_noticias_ia(tema: str, maximo: int = 5) -> dict:
    """Busca noticias simuladas relacionadas con un tema de Inteligencia Artificial.

    Usá esta herramienta para encontrar noticias recientes relacionadas con
    el tema que te pidieron investigar. La búsqueda es por coincidencia de
    palabras dentro del título, el resumen y las etiquetas de cada noticia
    (sin distinguir mayúsculas de minúsculas). Si el tema es demasiado
    específico, está mal escrito o no tiene relación con IA, la búsqueda
    puede no encontrar ninguna noticia -- en ese caso, la lista de
    resultados viene vacía. Eso es información válida: hay que reportarlo
    honestamente, nunca inventar una noticia que no existe en la base.

    Args:
        tema: El tema o palabra clave a buscar (ej. "modelos", "regulación", "robótica").
        maximo: Cantidad máxima de noticias a devolver (por defecto 5).

    Returns:
        Un diccionario con "tema_buscado" y "noticias": una lista de hasta
        `maximo` noticias que matchean, cada una con "titular", "fuente",
        "fecha" y "resumen".
    """
    palabras_clave = _palabras_clave(tema)
    coincidencias = []
    for noticia in _NOTICIAS_IA:
        texto_buscable = " ".join(
            [noticia["titular"], noticia["resumen"], *noticia["etiquetas"]]
        ).lower()
        if any(palabra in texto_buscable for palabra in palabras_clave):
            coincidencias.append(noticia)

    seleccionadas = coincidencias[:maximo]
    return {
        "tema_buscado": tema,
        "noticias": [
            {
                "titular": n["titular"],
                "fuente": n["fuente"],
                "fecha": n["fecha"],
                "resumen": n["resumen"],
            }
            for n in seleccionadas
        ],
    }


TOOLS = [buscar_noticias_ia]

PROMPT_INVESTIGADOR = (
    "Sos un agente especializado ÚNICAMENTE en investigación de noticias "
    "de Inteligencia Artificial. Tu única tarea es usar la herramienta "
    "buscar_noticias_ia para encontrar noticias relacionadas con el tema "
    "que te indiquen. No analices el sentimiento de las noticias ni "
    "opines sobre ellas -- esa tarea es de otro agente especialista. Si "
    "la búsqueda no encuentra nada, decilo honestamente en tu respuesta, "
    "nunca inventes una noticia que la herramienta no devolvió."
)


def construir_agente_investigador(modelo):
    """Crea el agente ReAct del Investigador, con herramientas acotadas
    a una sola función: no tiene acceso a nada más que `buscar_noticias_ia`,
    así que no puede desviarse a hacer otra cosa."""
    return create_react_agent(modelo, tools=TOOLS, prompt=PROMPT_INVESTIGADOR)


# [Pre-entrega 7] Decorador de observabilidad: cuando LangSmith está
# activo, cada llamada a esta función aparece como un "span" propio
# dentro de la traza del job (con su entrada, salida, duración y los
# tokens de las llamadas al LLM que haga adentro). Si LangSmith no
# está configurado, el decorador no hace nada y no rompe nada.
@traceable(name="agente_investigador", run_type="chain", tags=["especialista", "Investigador"])
async def investigar(modelo, tema: str) -> ResultadoInvestigacion:
    """Punto de entrada que usa el nodo del grafo principal.

    Importante: le pasamos al agente SOLO la instrucción puntual (el
    tema a buscar) -- no el historial completo de mensajes del
    orquestador. Esto es justo lo que evita la "Contaminación de
    Contexto" que la consigna marca como error común: el especialista no
    necesita saber nada de la conversación previa entre el Supervisor y
    el Analista, solo necesita saber QUÉ tiene que buscar.
    """
    agente = construir_agente_investigador(modelo)
    entrada = {"messages": [HumanMessage(content=f"Buscá noticias sobre: {tema}")]}
    resultado = await agente.ainvoke(entrada)

    # El dato estructurado real no está en lo que "dice" el LLM en su
    # respuesta de texto -- está en el ToolMessage que devolvió la
    # herramienta. Lo leemos de ahí (la fuente de verdad), nunca
    # intentamos parsear el texto libre de la respuesta final del agente.
    mensajes_herramienta = [m for m in resultado["messages"] if isinstance(m, ToolMessage)]

    if not mensajes_herramienta:
        # El agente nunca llamó a la herramienta (puede pasar si el LLM
        # decide, por ejemplo, responder directamente). Lo tratamos como
        # "no se encontró nada" en vez de romper el grafo.
        return ResultadoInvestigacion(tema_buscado=tema, noticias=[])

    payload = json.loads(mensajes_herramienta[-1].content)

    try:
        return ResultadoInvestigacion(**payload)
    except ValidationError:
        # Si por algún motivo el payload no tiene la forma esperada,
        # preferimos devolver "no se encontró nada" antes que dejar
        # pasar un dato corrupto al resto del grafo.
        return ResultadoInvestigacion(tema_buscado=tema, noticias=[])
