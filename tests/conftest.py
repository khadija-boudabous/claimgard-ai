"""Fixtures partagées : environnement isolé du .env local et dépendances externes simulées."""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import models

from app.config import get_settings
from app.main import app

REQUIRED_ENV = {
    "QDRANT_URL": "http://qdrant.test:6333",
    "REDIS_URL": "redis://redis.test:6379",
    "LANGFUSE_HOST": "http://langfuse.test:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-super-secret",
}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Env explicite (prioritaire sur .env) + cache Settings vidé : mêmes résultats en local et en CI."""
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HEALTH_TIMEOUT_S", "0.05")
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def qdrant() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def redis() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def embedder() -> MagicMock:
    """Remplace HybridEmbedder : aucun modèle ONNX chargé ni téléchargé."""
    mock = MagicMock()
    mock.embed_query.return_value = ([0.1] * 384, models.SparseVector(indices=[1], values=[0.5]))
    return mock


@pytest.fixture
async def client(qdrant: AsyncMock, redis: AsyncMock, embedder: MagicMock) -> AsyncIterator[AsyncClient]:
    """ASGITransport ne déclenche pas le lifespan : on injecte directement les mocks dans app.state."""
    app.state.qdrant = qdrant
    app.state.redis = redis
    app.state.embedder = embedder
    app.state.reranker = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
