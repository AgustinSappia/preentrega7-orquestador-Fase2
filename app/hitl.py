"""
app/hitl.py
-----------
Human-in-the-loop (HITL): el punto del grafo donde el sistema se DETIENE
y espera que una persona apruebe antes de ejecutar una acción "crítica".

¿Cuál es la acción crítica en este proyecto?
    Publicar el reporte de sentimiento (nodo "Publicador" en graph.py).
    Es una herramienta con EFECTOS SECUNDARIOS: escribe en un canal
    compartido (la lista `reportes:publicados` en Redis) que otros
    sistemas/personas consumen. Una vez publicado, no se "des-publica".

¿Cuándo el sistema la considera crítica? (`evaluar_criticidad`)
    - Si el sentimiento general del reporte es NEGATIVO (publicar malas
      noticias sobre la industria tiene riesgo reputacional), o
    - si quien creó la tarea pidió explícitamente revisión humana
      (`forzar_aprobacion=True` en el POST /tasks).
    Si no es crítica, el reporte se publica directo, sin pausa.

¿Cómo se pausa el grafo de verdad? -> `interrupt()` de LangGraph
    Cuando el nodo llama a `interrupt(payload)`, LangGraph:
      1. guarda un checkpoint del estado completo (en Redis, gracias al
         checkpointer -- sin checkpointer, `interrupt` ni siquiera
         funciona),
      2. corta la ejecución y devuelve el `payload` a quien invocó el
         grafo (el worker), que marca el job como AWAITING_APPROVAL.
    Más tarde, el endpoint POST /tasks/{id}/approve reanuda el MISMO
    hilo (`thread_id = job_id`) con `Command(resume=decision)`: LangGraph
    carga el checkpoint desde Redis y vuelve a ejecutar este nodo desde el
    principio, pero esta vez `interrupt()` NO pausa: devuelve directamente
    la decisión humana. Por eso todo lo que está ANTES del `interrupt()`
    tiene que ser "puro" (sin efectos secundarios): se ejecuta dos veces.

El endpoint HTTP de aprobación también vive en este módulo (más abajo),
para que todo lo relacionado con HITL quede en un solo lugar.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.dependencies import obtener_grafo, obtener_jobs
from app.jobs import EstadoJob, Job, JobStore
from app.state import AgentState, DecisionHumana, RegistroAprobacion


# ---------------------------------------------------------------------------
# 1. Regla de criticidad (función pura: sin efectos secundarios)
# ---------------------------------------------------------------------------
def evaluar_criticidad(state: AgentState) -> tuple[bool, str]:
    """Devuelve (es_critica, motivo). El motivo queda registrado en el
    estado y se le muestra a la persona que tiene que aprobar."""
    analisis = state.get("analisis")

    if state.get("forzar_aprobacion"):
        return True, "Quien creó la tarea pidió revisión humana obligatoria (forzar_aprobacion=True)."

    if analisis is not None and analisis.sentimiento_general == "negativo":
        return True, (
            "El reporte tiene sentimiento general NEGATIVO "
            f"(score promedio {analisis.score_promedio}). Publicarlo tiene "
            "riesgo reputacional: requiere aprobación humana."
        )

    sentimiento = analisis.sentimiento_general if analisis else "desconocido"
    return False, f"Sentimiento '{sentimiento}' (no negativo): se publica sin intervención humana."


def _propuesta_de_publicacion(state: AgentState) -> dict:
    """Resumen de lo que se va a publicar, para que la persona decida con
    información concreta (y no tenga que leer el estado completo)."""
    analisis = state["analisis"]
    investigacion = state.get("investigacion")
    return {
        "topic": state["topic"],
        "sentimiento_general": analisis.sentimiento_general,
        "score_promedio": analisis.score_promedio,
        "cantidad_noticias": len(analisis.detalle),
        "titulares": [item.titular for item in analisis.detalle],
        "tema_buscado": investigacion.tema_buscado if investigacion else state["topic"],
    }


# ---------------------------------------------------------------------------
# 2. El nodo del grafo
# ---------------------------------------------------------------------------
async def nodo_aprobacion_humana(state: AgentState) -> dict:
    """Nodo "AprobacionHumana" del grafo.

    - Si la publicación NO es crítica: aprueba automáticamente (y lo deja
      registrado como "no requerida").
    - Si ES crítica: pausa el grafo con `interrupt()` hasta que llegue la
      decisión humana por la API.
    """
    es_critica, motivo = evaluar_criticidad(state)

    if not es_critica:
        return {
            "aprobacion": RegistroAprobacion(
                requerida=False, motivo=motivo, aprobado=True, revisor="sistema"
            )
        }

    # --- Acá el grafo se DETIENE (la primera vez que se ejecuta). ---
    # Lo que se pasa a interrupt() tiene que ser serializable (dict de
    # tipos simples), porque viaja dentro del checkpoint y por la API.
    respuesta = interrupt(
        {
            "tipo": "aprobacion_publicacion",
            "motivo": motivo,
            "propuesta": _propuesta_de_publicacion(state),
            "instrucciones": (
                "Enviar POST /tasks/{job_id}/approve con "
                '{"aprobado": true|false, "comentario": "...", "revisor": "..."}'
            ),
        }
    )
    # --- Cuando se reanuda, `respuesta` es lo que vino en Command(resume=...). ---

    # Validamos la decisión con Pydantic: aunque la API ya la valida, el
    # nodo no confía ciegamente en lo que le llega desde afuera.
    decision = DecisionHumana.model_validate(respuesta)
    return {
        "aprobacion": RegistroAprobacion(
            requerida=True,
            motivo=motivo,
            aprobado=decision.aprobado,
            comentario=decision.comentario,
            revisor=decision.revisor,
        )
    }


def rutear_tras_aprobacion(state: AgentState) -> str:
    """Arista condicional que sale de AprobacionHumana: solo se publica si
    quedó aprobado (automática o humanamente)."""
    aprobacion = state.get("aprobacion")
    if aprobacion is not None and aprobacion.aprobado:
        return "Publicador"
    return "FIN"


# ---------------------------------------------------------------------------
# 3. El endpoint HTTP de aprobación: POST /tasks/{job_id}/approve
# ---------------------------------------------------------------------------
router = APIRouter(tags=["human-in-the-loop"])


@router.post(
    "/tasks/{job_id}/approve",
    response_model=Job,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aprobar o rechazar una tarea pausada en AprobacionHumana",
)
async def aprobar_tarea(
    job_id: str,
    decision: DecisionHumana,
    background_tasks: BackgroundTasks,
    jobs: JobStore = Depends(obtener_jobs),
    grafo: CompiledStateGraph = Depends(obtener_grafo),
) -> Job:
    """Recibe la decisión humana y reanuda el grafo EN SEGUNDO PLANO.

    - 404 si el job no existe.
    - 409 si el job no está esperando aprobación (ya se aprobó, nunca la
      necesitó, terminó, falló...). La transición AWAITING_APPROVAL ->
      RUNNING es atómica (ver JobStore.transicionar): si llegan dos
      aprobaciones a la vez, solo una gana y la otra recibe 409.
    - 202 si se aceptó: el cliente vuelve a consultar GET /tasks/{id}
      para ver el resultado final (DONE o REJECTED).
    """
    # Import local para evitar un import circular (worker -> graph -> hitl).
    from app.worker import reanudar_job

    job = await jobs.transicionar(job_id, desde={EstadoJob.AWAITING_APPROVAL}, hacia=EstadoJob.RUNNING)
    if job is None:
        existente = await jobs.obtener(job_id)
        if existente is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job está en estado {existente.status.value}: solo se puede aprobar si está en AWAITING_APPROVAL",
        )

    background_tasks.add_task(reanudar_job, job_id, decision, grafo=grafo, jobs=jobs)
    return job
