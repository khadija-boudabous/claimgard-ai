from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import grpc
import httpx
import pytest
from httpx import AsyncClient
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.main import app


def _point(idx: int, score: float) -> models.ScoredPoint:
    payload = {
        "contract": "Assurance Auto",
        "section": "Garanties",
        "article": f"Article {idx}",
        "text": f"Clause {idx}",
    }
    return models.ScoredPoint(id=idx, version=0, score=score, payload=payload)


def _grpc_error(code: grpc.StatusCode) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata(), details="collection")


async def test_retrieve_returns_passages_and_latencies(client: AsyncClient, qdrant: AsyncMock) -> None:
    qdrant.query_points.return_value = SimpleNamespace(points=[_point(1, 0.812345), _point(2, 0.5)])

    response = await client.post("/retrieve", json={"query": "pare-brise fissuré", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0] == {
        "score": 0.8123,
        "contract": "Assurance Auto",
        "section": "Garanties",
        "article": "Article 1",
        "text": "Clause 1",
    }
    assert len(body["results"]) == 2
    assert body["rerank_ms"] == 0.0
    assert min(body["embedding_ms"], body["search_ms"]) >= 0
    assert body["total_ms"] >= body["search_ms"]

    kwargs = qdrant.query_points.call_args.kwargs
    assert kwargs["limit"] == 2
    assert kwargs["query"].fusion == models.Fusion.RRF


async def test_retrieve_with_reranker_reorders_and_truncates(client: AsyncClient, qdrant: AsyncMock) -> None:
    qdrant.query_points.return_value = SimpleNamespace(points=[_point(i, 0.5) for i in range(1, 4)])
    reranker = MagicMock()
    reranker.rerank.return_value = [0.1, 0.2, 0.9]  # le 3e candidat devient le meilleur
    app.state.reranker = reranker

    response = await client.post("/retrieve", json={"query": "vol", "top_k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["article"] for r in results] == ["Article 3", "Article 2"]
    assert results[0]["score"] == 0.9
    assert qdrant.query_points.call_args.kwargs["limit"] == 10  # max(rerank_candidates, top_k)


@pytest.mark.parametrize(
    "error",
    [
        _grpc_error(grpc.StatusCode.NOT_FOUND),
        UnexpectedResponse(404, "Not Found", b"", httpx.Headers()),
    ],
    ids=["grpc", "rest"],
)
async def test_retrieve_returns_503_when_collection_missing(
    client: AsyncClient, qdrant: AsyncMock, error: Exception
) -> None:
    qdrant.query_points.side_effect = error

    response = await client.post("/retrieve", json={"query": "carence dentaire"})

    assert response.status_code == 503
    assert "app.rag.ingest" in response.json()["detail"]


async def test_retrieve_propagates_other_grpc_errors(client: AsyncClient, qdrant: AsyncMock) -> None:
    qdrant.query_points.side_effect = _grpc_error(grpc.StatusCode.UNAVAILABLE)

    with pytest.raises(grpc.aio.AioRpcError):
        await client.post("/retrieve", json={"query": "carence dentaire"})


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "ok", "top_k": 11}, {"query": "ok", "top_k": 0}])
async def test_retrieve_rejects_invalid_payload(client: AsyncClient, payload: dict[str, Any]) -> None:
    response = await client.post("/retrieve", json=payload)
    assert response.status_code == 422
