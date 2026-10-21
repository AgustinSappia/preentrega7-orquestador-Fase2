# Capturas del dashboard de LangSmith

Las capturas de esta carpeta son la evidencia de observabilidad. Todas salen
de la misma corrida de `scripts/carga_concurrente.py` (5 peticiones
concurrentes), con `LANGSMITH_TRACING=true`.

| Archivo | Qué muestra | Dónde se encuentra en LangSmith |
|---|---|---|
| `01_lista_trazas.png` | Las trazas de las 5 ejecuciones concurrentes (más las reanudaciones de los HITL) | Proyecto → pestaña **Runs**, filtro `tag = preentrega7` |
| `02_arbol_traza.png` | El árbol de UNA traza: Supervisor → agente_investigador → agente_analista → AprobacionHumana → Publicador, con latencia y tokens de cada paso | Clic en una traza de la lista |
| `03_hitl_interrupt.png` | Una traza que termina en `AprobacionHumana` con el interrupt (job crítico) y su reanudación en el mismo Thread | Pestaña **Threads**, o filtro `metadata.job_id` |
| `04_costo_por_ejecucion.png` | **Costo por ejecución** (columna *Cost* / *Total Cost* de cada traza, calculada por LangSmith a partir de tokens de entrada y salida) | Lista de Runs con las columnas *Tokens* y *Cost* visibles |
| `05_latencia_p95.png` | **Latencia p95** de la corrida | Pestaña **Monitor** (o el panel de estadísticas del proyecto: *Latency P50 / P99*; en el gráfico de latencia elegí el percentil p95) |
| `06_tokens_por_nodo.png` | Qué nodo consume más tokens (normalmente el que tiene llamadas al LLM con más contexto) | Dentro del árbol de una traza, comparando los *spans* de cada agente |

Pasos para sacarlas: ver la sección "Prueba de carga y capturas" del README
principal.
