"""
scripts/carga_concurrente.py
----------------------------
Prueba de carga pedida por la consigna: lanza 5 peticiones CONCURRENTES
contra la API, hace polling hasta que todas terminan (aprobando
automáticamente las que piden aprobación humana) e imprime un resumen con
latencias.

Uso (con la API levantada):

    python scripts/carga_concurrente.py
    python scripts/carga_concurrente.py --url http://localhost:8000 --no-aprobar

Las métricas "oficiales" de la consigna (costo por ejecución y latencia
p95) se leen en el dashboard de LangSmith, que las calcula solo a partir
de las trazas. Este script calcula además la latencia vista desde el
cliente, como referencia para comparar.

Los 5 temas por defecto están elegidos para ejercitar los dos caminos:
    modelos, robótica, adopción -> sentimiento positivo -> se publican solos
    regulación, agentes         -> sentimiento negativo -> AWAITING_APPROVAL
"""

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

TEMAS_POR_DEFECTO = ["modelos", "regulación", "robótica", "adopción", "agentes"]
ESTADOS_FINALES = {"DONE", "REJECTED", "FAILED"}


def percentil(valores: list[float], p: float) -> float:
    """Percentil por el método "nearest-rank": el valor por debajo del cual
    queda el p% de las mediciones. Con pocas muestras (5), el p95 coincide
    con el máximo -- es esperable."""
    if not valores:
        return float("nan")
    ordenados = sorted(valores)
    k = max(0, math.ceil(p / 100 * len(ordenados)) - 1)
    return ordenados[k]


async def lanzar(cliente: httpx.AsyncClient, tema: str) -> dict:
    t0 = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"topic": tema})
    respuesta.raise_for_status()
    return {
        "tema": tema,
        "job_id": respuesta.json()["job_id"],
        "latencia_post_ms": round((time.perf_counter() - t0) * 1000, 1),
        "_t0": t0,
    }


async def seguir(cliente: httpx.AsyncClient, item: dict, aprobar: bool, intervalo: float, timeout: float) -> dict:
    """Polling de GET /tasks/{id} hasta un estado final."""
    while True:
        job = (await cliente.get(f"/tasks/{item['job_id']}")).json()
        estado = job["status"]

        if estado == "AWAITING_APPROVAL" and "hitl" not in item:
            item["hitl"] = True
            motivo = (job.get("aprobacion_pendiente") or {}).get("motivo", "")
            print(f"  ⏸  [{item['tema']}] pide aprobación humana: {motivo}")
            if not aprobar:
                item["estado"] = estado
                return item
            r = await cliente.post(
                f"/tasks/{item['job_id']}/approve",
                json={"aprobado": True, "comentario": "Aprobado por la prueba de carga", "revisor": "prueba-de-carga"},
            )
            print(f"  ✅ [{item['tema']}] aprobado (HTTP {r.status_code})")

        if estado in ESTADOS_FINALES:
            resultado = job.get("resultado") or {}
            analisis = resultado.get("analisis") or {}
            item.update(
                estado=estado,
                total_cliente_s=round(time.perf_counter() - item["_t0"], 2),
                duracion_servidor_s=job.get("duracion_segundos"),
                sentimiento=analisis.get("sentimiento_general"),
                publicado=bool(resultado.get("reporte_publicado")),
                error=job.get("error"),
            )
            return item

        if time.perf_counter() - item["_t0"] > timeout:
            item["estado"] = f"TIMEOUT_CLIENTE ({estado})"
            return item
        await asyncio.sleep(intervalo)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de carga: N peticiones concurrentes contra la API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--temas", nargs="+", default=TEMAS_POR_DEFECTO)
    parser.add_argument("--no-aprobar", action="store_true", help="No aprobar automáticamente los HITL.")
    parser.add_argument("--intervalo", type=float, default=0.5, help="Segundos entre consultas de polling.")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as cliente:
        try:
            salud = (await cliente.get("/health")).json()
        except httpx.HTTPError as exc:
            print(f"No se pudo conectar con la API en {args.url}: {exc}")
            return 1
        print(f"API: {salud}\n")

        print(f"Lanzando {len(args.temas)} peticiones CONCURRENTES...")
        inicio = time.perf_counter()
        items = await asyncio.gather(*[lanzar(cliente, t) for t in args.temas])
        for it in items:
            print(f"  → [{it['tema']}] job_id={it['job_id']}  (POST respondió en {it['latencia_post_ms']} ms)")

        print("\nHaciendo polling hasta que terminen...")
        items = await asyncio.gather(
            *[seguir(cliente, it, not args.no_aprobar, args.intervalo, args.timeout) for it in items]
        )
        total = time.perf_counter() - inicio

    # ---- Resumen --------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{'tema':<12} {'estado':<18} {'HITL':<5} {'sentim.':<9} {'publicó':<8} {'POST ms':>8} {'total s':>8} {'servidor s':>11}")
    print("-" * 96)
    for it in items:
        print(
            f"{it['tema']:<12} {it.get('estado', '?'):<18} {'sí' if it.get('hitl') else 'no':<5} "
            f"{str(it.get('sentimiento') or '-'):<9} {'sí' if it.get('publicado') else 'no':<8} "
            f"{it['latencia_post_ms']:>8} {str(it.get('total_cliente_s', '-')):>8} {str(it.get('duracion_servidor_s', '-')):>11}"
        )
    print("=" * 96)

    latencias_post = [it["latencia_post_ms"] for it in items]
    totales = [it["total_cliente_s"] for it in items if "total_cliente_s" in it]
    print(f"Tiempo total de la corrida: {total:.2f} s")
    print(f"Latencia de POST /tasks  -> máx {max(latencias_post):.1f} ms | p95 {percentil(latencias_post, 95):.1f} ms")
    if totales:
        print(f"Latencia extremo a extremo (cliente) -> p50 {percentil(totales, 50):.2f} s | p95 {percentil(totales, 95):.2f} s")
    fallidos = [it for it in items if it.get("estado") == "FAILED"]
    for it in fallidos:
        print(f"  ❌ [{it['tema']}] FAILED: {it.get('error')}")

    carpeta = Path(__file__).resolve().parent.parent / "resultados"
    carpeta.mkdir(exist_ok=True)
    archivo = carpeta / f"carga_{datetime.now():%Y%m%d_%H%M%S}.json"
    archivo.write_text(
        json.dumps([{k: v for k, v in it.items() if not k.startswith("_")} for it in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultados guardados en {archivo}")
    print(
        "Siguiente paso: abrí LangSmith -> proyecto de LANGSMITH_PROJECT -> filtrá por el tag 'preentrega7' "
        "y capturá la latencia p95 y el costo por ejecución (ver screenshots/README.md)."
    )
    return 1 if fallidos else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
