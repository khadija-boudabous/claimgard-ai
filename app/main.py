import asyncio
import time
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.config import get_settings
from app.rag.embedder import load_embedder
from app.rag.search import CollectionNotFoundError, Passage, hybrid_search


class ServiceStatus(BaseModel):
    status: Literal["up", "down"]
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, ServiceStatus]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class RetrieveResponse(BaseModel):
    results: list[Passage]
    embedding_ms: float
    search_ms: float
    total_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.qdrant = AsyncQdrantClient(url=str(settings.qdrant_url))
    app.state.redis = Redis.from_url(str(settings.redis_url))
    # Chargement unique des modèles au démarrage : aucune requête ne paie le cold start
    app.state.embedder = await asyncio.to_thread(load_embedder, settings)
    yield
    await app.state.qdrant.close()
    await app.state.redis.aclose()


app = FastAPI(title="ClaimGuard AI", lifespan=lifespan)


async def _probe(check: Awaitable[Any], timeout_s: float) -> ServiceStatus:
    """Exécute une sonde réseau avec timeout et mesure sa latence."""
    start = time.perf_counter()
    try:
        await asyncio.wait_for(check, timeout=timeout_s)
        state, error = "up", None
    except Exception as exc:  # noqa: BLE001 — toute panne = service down
        state, error = "down", type(exc).__name__  # pas de message : évite de fuiter URLs/secrets
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ServiceStatus(status=state, latency_ms=latency_ms, error=error)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request, response: Response) -> HealthResponse:
    timeout_s = get_settings().health_timeout_s
    qdrant, redis = await asyncio.gather(
        _probe(request.app.state.qdrant.get_collections(), timeout_s),
        _probe(request.app.state.redis.ping(), timeout_s),
    )
    services = {"qdrant": qdrant, "redis": redis}
    all_up = all(s.status == "up" for s in services.values())
    if not all_up:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if all_up else "degraded", services=services)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest, request: Request) -> RetrieveResponse:
    settings = get_settings()
    start = time.perf_counter()
    try:
        result = await hybrid_search(
            request.app.state.qdrant,
            request.app.state.embedder,
            settings.qdrant_collection,
            body.query,
            body.top_k,
            settings.prefetch_limit,
        )
    except CollectionNotFoundError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Index absent : lancez `python -m app.rag.ingest`.",
        ) from None
    return RetrieveResponse(
        results=result.passages,
        embedding_ms=result.embedding_ms,
        search_ms=result.search_ms,
        total_ms=round((time.perf_counter() - start) * 1000, 2),
    )
