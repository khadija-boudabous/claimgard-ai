"""Recherche hybride dense + BM25 fusionnée par RRF côté Qdrant (un seul aller-retour)."""

import asyncio
import time

import grpc
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.embedder import DENSE_VECTOR, SPARSE_VECTOR, HybridEmbedder


class CollectionNotFoundError(RuntimeError):
    """La collection n'existe pas : l'ingestion n'a pas été lancée."""


class Passage(BaseModel):
    score: float  # score RRF : rang fusionné, pas une similarité
    contract: str
    section: str
    article: str
    text: str


class SearchResult(BaseModel):
    passages: list[Passage]
    embedding_ms: float
    search_ms: float


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _is_not_found(exc: Exception) -> bool:
    """Collection absente, que le client parle REST (404) ou gRPC (NOT_FOUND)."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404
    return isinstance(exc, grpc.aio.AioRpcError) and exc.code() == grpc.StatusCode.NOT_FOUND


async def hybrid_search(
    client: AsyncQdrantClient,
    embedder: HybridEmbedder,
    collection: str,
    query: str,
    top_k: int,
    prefetch_limit: int,
) -> SearchResult:
    start = time.perf_counter()
    # Inférence ONNX CPU-bound : hors event loop pour ne pas bloquer les autres requêtes
    dense, sparse = await asyncio.to_thread(embedder.embed_query, query)
    embedding_ms = _elapsed_ms(start)

    start = time.perf_counter()
    try:
        response = await client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE_VECTOR, limit=prefetch_limit),
                models.Prefetch(query=sparse, using=SPARSE_VECTOR, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    except (UnexpectedResponse, grpc.aio.AioRpcError) as exc:
        if _is_not_found(exc):
            raise CollectionNotFoundError(collection) from exc
        raise
    search_ms = _elapsed_ms(start)

    passages = [
        Passage(score=round(p.score, 4), **{k: p.payload[k] for k in Passage.model_fields if k != "score"})
        for p in response.points
        if p.payload
    ]
    return SearchResult(passages=passages, embedding_ms=embedding_ms, search_ms=search_ms)
