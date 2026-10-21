# Capturas del dashboard de LangSmith

Todas las capturas son de la misma corrida de `scripts/carga_concurrente.py`:
5 peticiones concurrentes del 21/10/2026 a las 11:50 (hora de Argentina),
con `gpt-4o-mini`. Están en el proyecto `preentrega7-orquestador` de
LangSmith.

| Archivo | Qué muestra |
|---|---|
| `01_lista_trazas_latencia_tokens_costo.jpg` | Las 7 trazas de la corrida: 5 `orquestador-ejecucion` (una por petición) y 2 `orquestador-reanudacion` (los jobs que pasaron por aprobación humana). Cada fila tiene su **latencia, tokens y costo**. A la izquierda se ven los tags y run names usados para filtrar |
| `02_arbol_traza_waterfall.jpg` | Vista *Waterfall* de la traza de `modelos` (11.26 s, 2.966K tokens, $0.0007). Se ve **dónde se concentra la latencia**: Supervisor, Investigador (con sus 2 llamadas a ChatOpenAI y la herramienta `buscar_noticias_ia`), Analista (2 llamadas y `analizar_sentimiento_noticias`), decisión final del Supervisor, AprobacionHumana y Publicador |
| `03_hitl_interrupt_y_reanudacion.jpg` | El **Human-in-the-loop** de `regulación`, agrupado en un mismo *Thread* (2 turnos). En el turno 1, `AprobacionHumana` termina en **Interrupted Run / GraphInterrupt**, con el motivo "sentimiento general NEGATIVO". En el turno 2, la reanudación (0.04 s) pasa por AprobacionHumana, `rutear_tras_aprobacion` y **Publicador** |
| `04_costo_por_ejecucion_threads.jpg` | Vista *Threads*: un hilo por job, con turnos, latencia P50/P99, tokens y **costo por ejecución** (los jobs con HITL muestran 2 turnos) |
| `05_latencia_por_ejecucion.jpg` | Las columnas Latency, Tokens, Cost y Tags de las 7 trazas. Con esas 5 latencias se calcula la **latencia p95** (ver el README principal) |
| `06a_tokens_investigador_llamada1.jpg` | 1ª llamada al LLM del Investigador (decide llamar a la herramienta): 403 tokens (384 de entrada / 19 de salida) |
| `06b_tokens_investigador_llamada2.jpg` | 2ª llamada al LLM del Investigador (redacta la respuesta con el resultado de la herramienta en el contexto): **740 tokens** (577 / 163). Es la llamada más cara de las medidas |
| `06c_tokens_analista.jpg` | Llamada al LLM del Analista: 632 tokens (480 / 152) |

La interpretación de estos números está en la sección "Lectura del dashboard
de LangSmith" del README principal.
