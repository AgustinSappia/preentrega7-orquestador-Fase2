# Pre-entrega 7: API de producción y monitoreo activo

API REST con **FastAPI** que expone el orquestador multi-agente de la
Pre-entrega 6. Ese orquestador tiene un Supervisor, un Investigador y un
Analista, y trabaja sobre noticias de IA. Esta versión le suma:

- **Endpoints asíncronos.** `POST /tasks` encola la tarea y devuelve un `job_id` al instante, sin bloquear el event loop.
- **Redis.** Guarda el estado de cada job (`PENDING → RUNNING → DONE / FAILED …`) y los **checkpoints de LangGraph** (`AsyncRedisSaver`).
- **Observabilidad con LangSmith.** Cada ejecución envía una traza con latencia, tokens y costo de cada nodo.
- **Human-in-the-loop.** Antes de **publicar** un reporte (una acción con efectos secundarios), el grafo se pausa con `interrupt()` si la tarea es crítica. Recién continúa cuando llega `POST /tasks/{id}/approve`.
- **Prueba de carga.** El script `scripts/carga_concurrente.py` lanza 5 peticiones concurrentes.

---

## 1. Arquitectura

```
                       ┌──────────────────────────── FastAPI (un proceso) ─────────────────────────────┐
 cliente ── POST /tasks ──► crea job PENDING en Redis ──► responde 202 {job_id}  (no espera al grafo)     │
                       │         │                                                                     │
                       │         └─ BackgroundTask: worker.ejecutar_job(job_id)                        │
                       │               RUNNING ─► grafo.ainvoke(thread_id=job_id) ─► DONE / FAILED     │
                       │                                         │                 └► AWAITING_APPROVAL│
 cliente ── GET /tasks/{id} ──► lee el job en Redis (polling)    │                                     │
 humano ─── POST /tasks/{id}/approve ──► AWAITING_APPROVAL→RUNNING (atómico) ─► worker.reanudar_job    │
                       └─────────────────────────────────────────┼─────────────────────────────────────┘
                                                                 │
          ┌──────────────── Redis 8 ───────────────┐             │  trazas (async, en lotes)
          │ job:{id}              estado del job    │             └──────────────► LangSmith
          │ checkpoint:*          checkpoints grafo │                             (latencia, tokens,
          │ reportes:publicados   efecto secundario │                              costo, p95)
          └─────────────────────────────────────────┘
```

### El grafo (generado con `grafo.get_graph().draw_mermaid()`)

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	Supervisor(Supervisor)
	Investigador(Investigador)
	Analista(Analista)
	AprobacionHumana(AprobacionHumana)
	Publicador(Publicador)
	__end__([<p>__end__</p>]):::last
	Analista --> Supervisor;
	AprobacionHumana -.-> Publicador;
	AprobacionHumana -. &nbsp;FIN&nbsp; .-> __end__;
	Investigador --> Supervisor;
	Supervisor -.-> Analista;
	Supervisor -.-> AprobacionHumana;
	Supervisor -.-> Investigador;
	Supervisor -. &nbsp;FIN&nbsp; .-> __end__;
	__start__ --> Supervisor;
	Publicador --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Respecto de la Pre-entrega 6, cambia una sola cosa del flujo. Cuando el
Supervisor decide `FINALIZAR` y ya hay un análisis suficiente, el grafo no
termina: pasa por **AprobacionHumana** y después por **Publicador**. Si se
cortó por el contador de pasos y no hay análisis útil, termina sin publicar.

### Estructura del repositorio

```
preentrega7/
├── app/
│   ├── main.py            # FastAPI: lifespan (Redis + checkpointer), POST/GET /tasks, /reportes, /health
│   ├── graph.py           # orquestador M6 + nodos AprobacionHumana/Publicador + checkpointer_redis()
│   ├── worker.py          # corre/reanuda el grafo en segundo plano; cualquier excepción → FAILED
│   ├── jobs.py            # JobStore: estado de los jobs en Redis (async, TTL, transición atómica)
│   ├── observability.py   # LangSmith: validación al arrancar, metadata/tags por job, flush al apagar
│   ├── hitl.py            # regla de criticidad + nodo con interrupt() + POST /tasks/{id}/approve
│   ├── dependencies.py    # dependencias de FastAPI (JobStore y grafo desde app.state)
│   ├── state.py           # AgentState + artefactos Pydantic + lista blanca de tipos del checkpointer
│   └── agents/            # Investigador y Analista (Pre-entrega 6, con @traceable)
├── scripts/carga_concurrente.py   # 5 peticiones concurrentes + resumen de latencias
├── tests/                 # 58 tests, sin API key ni Redis (LLM falso + fakeredis + InMemorySaver)
├── screenshots/           # capturas de LangSmith (ver screenshots/README.md)
├── Dockerfile · docker-compose.yml · requirements.txt · .env.example · pytest.ini
└── README.md
```

---

## 2. Cómo levantarlo

### Requisitos

- Docker Desktop (para Redis 8 y, si querés, también la API).
- Una API key de OpenAI (o Anthropic).
- Una API key de LangSmith. Se crea gratis en <https://smith.langchain.com>, en **Settings → API Keys**.

```bash
cp .env.example .env      # completar OPENAI_API_KEY y LANGSMITH_API_KEY
```

### Opción A: todo con Docker Compose (recomendada)

```bash
docker compose up --build
```

Esto levanta Redis 8 y la API en <http://localhost:8000>. La documentación
interactiva (Swagger) queda en <http://localhost:8000/docs>.

### Opción B: Redis en Docker y la API en tu máquina

```bash
docker compose up -d redis             # solo Redis 8
python -m venv venv
venv\Scripts\activate                  # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> **Importante: hace falta Redis 8, no Redis 7.** El checkpointer de
> LangGraph (`AsyncRedisSaver`) usa los módulos de búsqueda y JSON de Redis,
> que vienen incluidos desde Redis 8. Por eso el `docker-compose.yml` usa la
> imagen `redis:8`. Con un Redis 7 sin esos módulos, la API falla al
> arrancar con `unknown command 'FT.INFO'`.

Para comprobar que todo está bien:

```bash
curl http://localhost:8000/health
# {"status":"ok","redis":true,"observabilidad":{"plataforma":"langsmith","activo":true,"proyecto":"preentrega7-orquestador"}}
```

---

## 3. Uso de la API

| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/tasks` | Crea y encola una tarea. Devuelve **202** con el `job_id` |
| `GET` | `/tasks/{job_id}` | Devuelve el estado del job (para hacer polling) |
| `GET` | `/tasks` | Lista los últimos jobs |
| `POST` | `/tasks/{job_id}/approve` | Envía la decisión humana (`{"aprobado": true/false, "comentario", "revisor"}`) |
| `GET` | `/reportes` | Lista los reportes publicados (el efecto secundario) |
| `GET` | `/health` | Muestra el estado de Redis y de LangSmith |

### Ejemplo: tarea crítica con aprobación humana

```bash
# 1) Encolar: responde enseguida con el job_id
curl -X POST localhost:8000/tasks -H "Content-Type: application/json" -d '{"topic": "regulación"}'
# {"job_id":"ec97562c-...","status":"PENDING","links":{...}}

# 2) Consultar: el sentimiento es negativo, así que queda esperando aprobación
curl localhost:8000/tasks/ec97562c-...
# {"status":"AWAITING_APPROVAL","aprobacion_pendiente":{"motivo":"El reporte tiene sentimiento general NEGATIVO ...","propuesta":{...}}, ...}

# 3) Aprobar (o rechazar con "aprobado": false)
curl -X POST localhost:8000/tasks/ec97562c-.../approve -H "Content-Type: application/json" \
     -d '{"aprobado": true, "comentario": "OK", "revisor": "rayo"}'

# 4) Consultar de nuevo: DONE, y el reporte aparece en /reportes
curl localhost:8000/tasks/ec97562c-...
```

> En PowerShell, `curl` es un alias de otro comando. Usá `curl.exe`, o
> directamente Swagger en `/docs`, que es lo más cómodo.

### Estados de un job

| Estado | Significado |
|---|---|
| `PENDING` | Se encoló y todavía no arrancó |
| `RUNNING` | El grafo está corriendo, o reanudándose después de una aprobación |
| `AWAITING_APPROVAL` | Está pausado en `AprobacionHumana`. El campo `aprobacion_pendiente` explica por qué y qué se va a publicar |
| `DONE` | Terminó. Publicó, o no había nada que publicar |
| `REJECTED` | La persona que revisaba rechazó la publicación |
| `FAILED` | Hubo una excepción o un timeout en el worker. El campo `error` dice qué pasó |

---

## 4. Decisiones de diseño

### Asincronía: cómo se evita bloquear el event loop

- `POST /tasks` hace **una sola escritura** en Redis y responde **202**. El grafo se ejecuta después, como `BackgroundTask`.
- Todo lo que espera algo externo es `async`:
  - Redis, con `redis.asyncio`;
  - el checkpointer, con `AsyncRedisSaver`;
  - el grafo, con `ainvoke`, que a su vez llama al LLM de forma asíncrona.

  No hay ninguna llamada de red síncrona dentro de un endpoint ni del worker. Las herramientas de los agentes son funciones síncronas, pero muy cortas, y LangChain las corre en un thread aparte. Mientras una tarea espera al LLM, el event loop atiende otros pedidos.
- Hay **un solo grafo compilado** para toda la API. Los jobs concurrentes no se mezclan porque cada uno usa su propio `thread_id = job_id`. El test `test_cinco_jobs_concurrentes_con_el_mismo_grafo_no_se_mezclan` lo verifica.
- **Worker:** se usa el patrón de BackgroundTask simple, que la consigna acepta para esta etapa. Para escalar a varios servidores, el siguiente paso sería un worker separado (Arq o Celery) que lea de una cola en Redis. No se implementó para no sumar otra pieza que levantar.

### Manejo de errores en background ("error común" de la consigna)

`worker.ejecutar_job` y `worker.reanudar_job` envuelven todo en
`try/except Exception` y además tienen un **timeout**
(`TIMEOUT_JOB_SEGUNDOS`, 300 s por defecto). Cualquier falla deja el job en
**`FAILED`** con el mensaje de error, y el cliente deja de hacer polling.
Esto lo cubren tests de excepción en el grafo, de timeout y de error al
reanudar.

### Persistencia con Redis

En Redis se guardan dos cosas distintas:

| Clave | Para qué | Quién la escribe |
|---|---|---|
| `job:{id}` | Estado del job que ve el cliente (JSON, vence a los 7 días) | `JobStore` (`app/jobs.py`) |
| `checkpoint:*`, `checkpoint_write:*` | "Fotos" del estado del grafo, nodo por nodo (vencen a las 24 h) | `AsyncRedisSaver` (`app/graph.py`) |
| `reportes:publicados` | El "canal" donde se publican los reportes (efecto secundario) | nodo `Publicador` |

Gracias a los checkpoints en Redis, un job pausado sobrevive a un reinicio de
la API. El test `test_reanudar_con_un_grafo_nuevo_mismo_checkpointer`
simula ese caso.

Se pueden inspeccionar las claves así:

```bash
docker compose exec redis redis-cli --scan --pattern 'job:*'
docker compose exec redis redis-cli GET job:<id>
docker compose exec redis redis-cli LRANGE reportes:publicados 0 -1
```

Otros detalles de la persistencia:

- **Aprobaciones simultáneas.** La transición `AWAITING_APPROVAL → RUNNING` es atómica: usa `WATCH/MULTI/EXEC`, que es concurrencia optimista. Si llegan dos aprobaciones a la vez, solo una reanuda el grafo y la otra recibe **409**. Lo probé contra un Redis real lanzando 10 aprobaciones simultáneas, y ganó exactamente 1.
- **Lista blanca del checkpointer.** El checkpointer solo puede reconstruir las clases propias listadas en `TIPOS_SERIALIZABLES` (`app/state.py`). Reconstruir objetos a partir de bytes guardados afuera es un punto sensible de seguridad, y LangGraph va a bloquear por defecto las clases no registradas.

### Human-in-the-loop

- **La acción crítica** es el nodo `Publicador`: escribe el reporte en `reportes:publicados`, un efecto secundario que no se puede deshacer.
- **Cuándo es crítica** (`evaluar_criticidad` en `app/hitl.py`): cuando el sentimiento general es **negativo**, por el riesgo reputacional de publicar malas noticias, o cuando el cliente envía `"forzar_aprobacion": true`. En cualquier otro caso se publica sin pausa, y queda registrado `aprobacion.requerida = false`.
- **Cómo se pausa.** `interrupt(payload)` guarda un checkpoint y corta la ejecución. El worker ve `__interrupt__` en el resultado y marca el job como `AWAITING_APPROVAL`.
- **Cómo se reanuda.** `Command(resume=decision)` en el mismo `thread_id`. LangGraph vuelve a ejecutar el nodo desde el principio, y esta vez `interrupt()` devuelve la decisión. Por eso todo lo que está antes del `interrupt()` es puro, sin efectos secundarios: se ejecuta dos veces.
- **Validación.** La decisión se valida con Pydantic (`DecisionHumana`) dos veces: en el endpoint y otra vez dentro del nodo.

### Observabilidad (LangSmith)

- **Activación.** Con `LANGSMITH_TRACING=true`, LangGraph envía cada `ainvoke` como una traza, sin tocar los nodos.
- **`@traceable`** en `investigar()` y `analizar()`, las funciones que llaman al LLM. Así aparecen como *spans* con nombre propio: `agente_investigador` y `agente_analista`.
- **Etiquetas por ejecución** (`config_de_ejecucion`):
  - `run_name`: `orquestador-ejecucion` u `orquestador-reanudacion`;
  - `tags`: `preentrega7` y la etapa;
  - `metadata`: `job_id`, `thread_id`, modelo.

  Con `metadata.thread_id`, LangSmith agrupa la ejecución inicial y la reanudación del HITL en el mismo *Thread*.
- **Validación al arrancar.** Si falta la API key, se apaga el tracing con un aviso en vez de fallar en cada traza.
- **Al apagar.** `wait_for_all_tracers()` envía las trazas que hayan quedado pendientes.

### Búsqueda más robusta (mejora sobre la Pre-entrega 6)

`buscar_noticias_ia` ahora busca por palabras clave. Antes exigía que la frase
completa apareciera textual en la noticia. Con un LLM real, que a veces pasa
`"regulación de la IA"` en lugar de `"regulación"`, la versión anterior
devolvía 0 resultados.

---

## 5. Prueba de carga y capturas

Con la API levantada y LangSmith activo:

```bash
python scripts/carga_concurrente.py              # aprueba automáticamente los HITL
python scripts/carga_concurrente.py --no-aprobar # deja los HITL pausados (para capturarlos)
```

El script lanza 5 `POST /tasks` **a la vez**, con los temas `modelos`,
`regulación`, `robótica`, `adopción` y `agentes`:

- 3 tienen sentimiento positivo y se publican solos.
- 2 tienen sentimiento negativo y pasan por aprobación humana.

Después hace polling hasta que terminan, imprime una tabla con latencias
(incluido el p95 visto desde el cliente) y guarda los resultados en
`resultados/carga_*.json`.

Cómo leer el dashboard de LangSmith:

1. Entrá al proyecto `preentrega7-orquestador` y filtrá por `tag = preentrega7`.
2. **Latencia p95:** pestaña **Monitor**, gráfico de latencia en percentil p95.
3. **Costo por ejecución:** columna *Cost* de cada traza. LangSmith la calcula a partir de los tokens de entrada y salida.
4. **Nodo con más tokens:** abrí una traza y compará los *spans* `agente_investigador`, `agente_analista` y la decisión del Supervisor.

El detalle de cada captura está en [`screenshots/README.md`](screenshots/README.md).

### Resultado real de la prueba de carga

> **PENDIENTE:** pegar acá la salida real de `python scripts/carga_concurrente.py`
> corrida contra la API con LLM real y LangSmith activo, junto con la lectura
> del dashboard: p95, costo por ejecución y nodo que más tokens consume.

### Verificación hecha en el entorno de desarrollo (sin API key)

Para validar la mecánica sin gastar en el LLM, se levantó la app real (con su
`lifespan`, uvicorn y un Redis real para los jobs) con dos reemplazos:

- un **LLM falso** que responde según el contexto, sin memoria entre llamadas;
- un **checkpointer en memoria**, porque ese entorno solo tenía Redis 7.

Esta fue la salida real del script de carga:

```
Lanzando 5 peticiones CONCURRENTES...
  → [modelos] job_id=de023523-...  (POST respondió en 9.2 ms)
  → [regulación] job_id=ec97562c-...  (POST respondió en 58.3 ms)
  → [robótica] job_id=d425bdf6-...  (POST respondió en 58.7 ms)
  → [adopción] job_id=af79b369-...  (POST respondió en 61.9 ms)
  → [agentes] job_id=0b04f178-...  (POST respondió en 43.4 ms)

Haciendo polling hasta que terminen...
  ⏸  [regulación] pide aprobación humana: El reporte tiene sentimiento general NEGATIVO (score promedio 0.05). ...
  ⏸  [agentes] pide aprobación humana: El reporte tiene sentimiento general NEGATIVO (score promedio 0.05). ...
  ✅ [regulación] aprobado (HTTP 202)
  ✅ [agentes] aprobado (HTTP 202)

tema         estado             HITL  sentim.   publicó   POST ms  total s  servidor s
modelos      DONE               no    positivo  sí            9.2     0.53       0.325
regulación   DONE               sí    negativo  sí           58.3     0.74       0.484
robótica     DONE               no    positivo  sí           58.7     0.56       0.298
adopción     DONE               no    positivo  sí           61.9     0.53       0.285
agentes      DONE               sí    negativo  sí           43.4     0.75         0.5
Tiempo total de la corrida: 0.75 s
Latencia de POST /tasks  -> máx 61.9 ms | p95 61.9 ms
Latencia extremo a extremo (cliente) -> p50 0.56 s | p95 0.75 s
```

---

## 6. Tests automatizados (no necesitan API key, Redis ni Docker)

```bash
pytest -v
```

Son **58 tests**:

| Archivo | Qué cubre |
|---|---|
| `test_graph.py` | Regla de criticidad; tarea no crítica que publica sin pausa; tarea crítica que se pausa y se publica al aprobar; rechazo; reanudación con **otra instancia** del grafo (simula un reinicio); decisión humana inválida; refinamiento y corte por contador (heredados de la Pre-entrega 6) |
| `test_jobs.py` | `JobStore` sobre **fakeredis**: crear y leer, TTL, tiempos del ciclo de vida, transición atómica, **doble aprobación simultánea**, publicación, listado |
| `test_worker.py` | `DONE`, `AWAITING_APPROVAL → DONE`, `REJECTED`; **excepción → FAILED**; **timeout → FAILED**; error al reanudar → FAILED; 5 jobs concurrentes con el mismo grafo sin mezclarse |
| `test_api.py` | Endpoints con `httpx.ASGITransport`: 202, 404, 422, FAILED visible por HTTP, flujo HITL completo, rechazo, 409 al aprobar dos veces o un job que no espera aprobación |
| `test_no_bloqueante.py` | **Servidor uvicorn real**: 5 POST concurrentes contra un grafo que tarda 1 s. Cada POST responde en menos de 0.5 s, `/health` responde mientras corren, y las 5 terminan en unos 1 s, no en 5 |
| `test_research_agent.py`, `test_analyst_agent.py`, `test_state.py` | Agentes y modelos de la Pre-entrega 6, más la búsqueda por palabras clave |

Cómo se reemplaza cada pieza externa en los tests:

- **El LLM:** `ModeloFalso` da respuestas en orden, y `ModeloSegunContexto` responde según el último mensaje, lo que lo hace seguro con tareas concurrentes.
- **Redis:** `fakeredis`, un Redis en memoria.
- **El checkpointer:** `InMemorySaver`, con el **mismo serializador y la misma lista blanca** que el de Redis.

---

## 7. Cumplimiento de la rúbrica

| Criterio (peso) | Cómo se cumple | Dónde |
|---|---|---|
| **Orquestación asíncrona y API (30%)** | `POST /tasks` responde 202 con `job_id` sin esperar al grafo. BackgroundTask como worker. Todo async (`redis.asyncio`, `AsyncRedisSaver`, `ainvoke`). Probado con un servidor real y 5 peticiones concurrentes | `app/main.py`, `app/worker.py`, `tests/test_no_bloqueante.py` |
| **Persistencia con Redis (25%)** | Estado del job en Redis con TTL, incluido el paso a **FAILED** ante excepción o timeout. Checkpoints de LangGraph con `AsyncRedisSaver`. Transición atómica para aprobar. Redis 8 con AOF en Docker Compose | `app/jobs.py`, `app/graph.py` (`checkpointer_redis`), `app/worker.py`, `docker-compose.yml` |
| **Observabilidad (25%)** | LangSmith con trazas de cada nodo, `@traceable` en los agentes, metadata y tags por job, threads para el HITL. Capturas de trazas, costo por ejecución y p95 de las 5 peticiones concurrentes | `app/observability.py`, `app/agents/*`, `screenshots/` |
| **Human-in-the-loop (15%)** | Nodo `AprobacionHumana` con `interrupt()` antes de una acción con efectos secundarios, regla de criticidad y `POST /tasks/{id}/approve` con reanudación vía `Command(resume=...)` | `app/hitl.py`, `app/graph.py` |
| **Documentación y estructura (5%)** | Este README: cómo levantar Redis y la API, cómo lanzar las 5 peticiones, `requirements.txt`, `.env.example`, Docker Compose | `README.md`, `requirements.txt`, `docker-compose.yml` |
