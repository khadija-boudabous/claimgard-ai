"""Retrieval en deux étages : hybride dense + BM25 fusionné par RRF côté Qdrant, puis reranking."""

import asyncio
import time

import grpc
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.embedder import DENSE_VECTOR, SPARSE_VECTOR, HybridEmbedder
from app.rag.reranker import rerank


class CollectionNotFoundError(RuntimeError):
    """La collection n'existe pas : l'ingestion n'a pas été lancée."""


class Passage(BaseModel):
    score: float  # score cross-encoder si reranking, sinon score RRF (rang fusionné)
    contract: str
    section: str
    article: str
    text: str

    def context_text(self) -> str:
        """Même préfixe contextuel qu'à l'ingestion : situe la clause dans son contrat."""
        header = " — ".join(p for p in (self.contract, self.section, self.article) if p)
        return f"[{header}]\n{self.text}"


class SearchResult(BaseModel):
    passages: list[Passage]
    embedding_ms: float
    search_ms: float
    rerank_ms: float


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _is_not_found(exc: Exception) -> bool:
    """Collection absente, que le client parle REST (404) ou gRPC (NOT_FOUND)."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404
    return isinstance(exc, grpc.aio.AioRpcError) and exc.code() == grpc.StatusCode.NOT_FOUND


async def _query_qdrant(
    client: AsyncQdrantClient,
    collection: str,
    dense: list[float],
    sparse: models.SparseVector,
    limit: int,
    prefetch_limit: int,
) -> list[Passage]:
    try:
        response = await client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE_VECTOR, limit=prefetch_limit),
                models.Prefetch(query=sparse, using=SPARSE_VECTOR, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    except (UnexpectedResponse, grpc.aio.AioRpcError) as exc:
        if _is_not_found(exc):
            raise CollectionNotFoundError(collection) from exc
        raise
    return [
        Passage(score=round(p.score, 4), **{k: p.payload[k] for k in Passage.model_fields if k != "score"})
        for p in response.points
        if p.payload
    ]


def _rerank_passages(
    reranker: TextCrossEncoder, query: str, passages: list[Passage], top_k: int
) -> list[Passage]:
    ranked = rerank(reranker, query, [p.context_text() for p in passages], top_k)
    return [passages[i].model_copy(update={"score": round(score, 4)}) for i, score in ranked]


async def hybrid_search(
    client: AsyncQdrantClient,
    embedder: HybridEmbedder,
    reranker: TextCrossEncoder | None,
    collection: str,
    query: str,
    top_k: int,
    prefetch_limit: int,
    rerank_candidates: int,
) -> SearchResult:
    start = time.perf_counter()
    # Inférence ONNX CPU-bound : hors event loop pour ne pas bloquer les autres requêtes
    dense, sparse = await asyncio.to_thread(embedder.embed_query, query)
    embedding_ms = _elapsed_ms(start)

    start = time.perf_counter()
    limit = max(rerank_candidates, top_k) if reranker else top_k
    passages = await _query_qdrant(client, collection, dense, sparse, limit, prefetch_limit)
    search_ms = _elapsed_ms(start)

    rerank_ms = 0.0
    if reranker and passages:
        start = time.perf_counter()
        passages = await asyncio.to_thread(_rerank_passages, reranker, query, passages, top_k)
        rerank_ms = _elapsed_ms(start)

    return SearchResult(
        passages=passages, embedding_ms=embedding_ms, search_ms=search_ms, rerank_ms=rerank_ms
    )
