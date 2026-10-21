"""
app/state.py
------------
Esquema de estado compartido del orquestador y los "artefactos"
(Pydantic) que cada agente especialista produce.

[Pre-entrega 7] Se suman dos artefactos nuevos para el flujo
Human-in-the-loop (`RegistroAprobacion`) y para el efecto secundario que
ese flujo protege (`ReportePublicado`), y la lista `TIPOS_SERIALIZABLES`
que necesita el checkpointer de Redis (ver más abajo).

Por qué esto resuelve dos de los puntos de la consigna a la vez:

1. "Estado Compartido Estructurado ... que permita rastrear qué agente
   ha contribuido con qué información": en vez de que todo viva mezclado
   dentro de `messages` (texto libre), cada especialista escribe su
   aporte en SU PROPIO campo tipado (`investigacion`, `analisis`). El
   supervisor puede entonces saber exactamente qué se hizo y qué falta
   mirando esos campos, sin tener que releer ni interpretar la
   conversación completa.

2. "Gestión de Estado y Validación": cada artefacto es un modelo
   Pydantic con sus propias reglas (scores entre 0 y 1, campos
   obligatorios) -- un dato que no cumple la forma esperada no puede
   llegar a formar parte del estado.
"""

from typing import List, Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Artefacto del Investigador
# ---------------------------------------------------------------------------
class NoticiaIA(BaseModel):
    """Una noticia simulada encontrada por el Investigador."""

    titular: str
    fuente: str
    fecha: str
    resumen: str = Field(min_length=10, description="Resumen breve de la noticia.")


class ResultadoInvestigacion(BaseModel):
    """Lo que el Investigador aporta al estado compartido."""

    tema_buscado: str
    noticias: List[NoticiaIA] = Field(default_factory=list)

    @property
    def encontro_resultados(self) -> bool:
        return len(self.noticias) > 0


# ---------------------------------------------------------------------------
# Artefacto del Analista
# ---------------------------------------------------------------------------
class AnalisisPorNoticia(BaseModel):
    """El sentimiento detectado para UNA noticia puntual."""

    titular: str
    sentimiento: Literal["positivo", "neutral", "negativo"]
    score: float = Field(ge=0.0, le=1.0, description="0 = muy negativo, 1 = muy positivo.")


class ResultadoAnalisis(BaseModel):
    """Lo que el Analista aporta al estado compartido."""

    sentimiento_general: Literal["positivo", "neutral", "negativo", "sin_datos"]
    score_promedio: float = Field(ge=0.0, le=1.0)
    detalle: List[AnalisisPorNoticia] = Field(default_factory=list)
    # False cuando no había noticias para analizar -- es la señal que usa
    # el Supervisor para decidir si hace falta volver a investigar en vez
    # de cerrar la conversación con una respuesta vacía.
    suficiente: bool


# ---------------------------------------------------------------------------
# [Pre-entrega 7] Artefactos del flujo Human-in-the-loop
# ---------------------------------------------------------------------------
class DecisionHumana(BaseModel):
    """Lo que manda la persona que revisa, vía POST /tasks/{id}/approve.

    Es lo que viaja dentro de `Command(resume=...)` para "despertar" al
    grafo que quedó pausado en el nodo de aprobación."""

    aprobado: bool
    comentario: str = ""
    revisor: str = "anonimo"


class RegistroAprobacion(BaseModel):
    """Queda guardado en el estado para dejar trazabilidad: si hizo falta
    aprobación, por qué, y qué se decidió."""

    requerida: bool
    motivo: str
    aprobado: bool
    comentario: str = ""
    revisor: str = ""


class ReportePublicado(BaseModel):
    """El resultado del efecto secundario "crítico": publicar el reporte
    de sentimiento (en este proyecto se publica en una lista de Redis,
    `reportes:publicados`, que hace de "canal de publicación")."""

    topic: str
    sentimiento_general: str
    score_promedio: float
    cantidad_noticias: int
    publicado_en: str  # fecha/hora ISO-8601 (UTC)


# ---------------------------------------------------------------------------
# Estado global del grafo
# ---------------------------------------------------------------------------
# Hereda de MessagesState (así conserva `messages` con su reducer
# add_messages, por si se quiere loguear la conversación de alto nivel),
# y le agrega los campos propios de este ejercicio -- ninguno de estos
# campos nuevos necesita un reducer: cada uno se REEMPLAZA entero cuando
# el nodo correspondiente corre, igual que en el ejercicio del pipeline
# Investigador/Revisor.
class AgentState(MessagesState):
    topic: str

    # Control del Supervisor
    next_agent: str
    task_completed: bool
    pasos: int  # contador anti "Supervisor Infinito"

    # Contribuciones de cada especialista, por separado y tipadas
    investigacion: Optional[ResultadoInvestigacion]
    analisis: Optional[ResultadoAnalisis]

    # [Pre-entrega 7] Human-in-the-loop y publicación
    forzar_aprobacion: bool  # el cliente puede exigir revisión humana sí o sí
    aprobacion: Optional[RegistroAprobacion]
    reporte: Optional[ReportePublicado]


# Techo de seguridad: si el Supervisor da más vueltas que esto sin llegar
# a una conclusión, se corta el ciclo y se responde con lo que se tenga
# hasta ese momento, en vez de seguir dando vueltas indefinidamente.
MAX_PASOS = 6


# ---------------------------------------------------------------------------
# [Pre-entrega 7] Tipos que el checkpointer tiene permitido reconstruir
# ---------------------------------------------------------------------------
# El checkpointer guarda el estado en Redis (serializado con msgpack) y lo
# vuelve a armar cuando el grafo se reanuda -- por ejemplo, después de
# que un humano aprueba. Reconstruir objetos Python a partir de bytes
# guardados afuera es un punto sensible de seguridad: LangGraph avisa
# (y en futuras versiones va a BLOQUEAR) cualquier clase que no esté
# explícitamente permitida. Esta lista es esa "lista blanca": solo estas
# clases propias se pueden reconstruir desde un checkpoint.
TIPOS_SERIALIZABLES = [
    ("app.state", nombre)
    for nombre in (
        "NoticiaIA",
        "ResultadoInvestigacion",
        "AnalisisPorNoticia",
        "ResultadoAnalisis",
        "RegistroAprobacion",
        "ReportePublicado",
    )
]

# La misma lista, en el formato que pide el modo JSON del serializador
# ("app", "state", "ResultadoAnalisis"). Hace falta porque el checkpointer
# de Redis guarda los checkpoints como JSON (no como msgpack): sin esta
# lista, al reanudar un job los artefactos vuelven como `dict` en vez de
# como objetos Pydantic, y el grafo falla con
# AttributeError: 'dict' object has no attribute 'sentimiento_general'.
TIPOS_SERIALIZABLES_JSON = [tuple(modulo.split(".")) + (clase,) for modulo, clase in TIPOS_SERIALIZABLES]
