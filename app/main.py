"""
FastAPI application entrypoint.

Mounts the query, training, and debug routers and initialises the database on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_debug import router as debug_router
from app.api.routes_query import router as query_router
from app.api.routes_train import router as train_router
from app.db.sqlite_client import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run init_db once at startup before the app begins serving requests."""
    init_db()
    yield


app = FastAPI(title="Local Text-to-SQL RAG", lifespan=lifespan)

app.include_router(query_router)
app.include_router(train_router)
app.include_router(debug_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
