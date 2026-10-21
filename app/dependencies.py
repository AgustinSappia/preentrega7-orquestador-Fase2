"""
app/dependencies.py
-------------------
"Dependencias" de FastAPI: funciones que los endpoints declaran como
parámetros con `Depends(...)` y que FastAPI resuelve solo en cada pedido.

El JobStore (Redis) y el grafo compilado se crean UNA sola vez al arrancar
la API (ver `lifespan` en main.py) y se guardan en `app.state`. Estas
funciones solo los van a buscar ahí. Ventaja: en los tests se cargan en
`app.state` versiones falsas (fakeredis, un grafo con LLM falso) sin tocar
ningún endpoint.
"""
from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from app.jobs import JobStore


def obtener_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def obtener_grafo(request: Request) -> CompiledStateGraph:
    return request.app.state.grafo
