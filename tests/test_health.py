import asyncio
from unittest.mock import AsyncMock

from httpx import AsyncClient


async def test_health_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    for name in ("qdrant", "redis"):
        assert body["services"][name]["status"] == "up"
        assert body["services"][name]["latency_ms"] >= 0
        assert body["services"][name]["error"] is None


async def test_health_degraded_when_redis_fails(client: AsyncClient, redis: AsyncMock) -> None:
    redis.ping.side_effect = ConnectionError("redis://user:password@redis.test refused")

    response = await client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["qdrant"]["status"] == "up"
    assert body["services"]["redis"] == {
        "status": "down",
        "latency_ms": body["services"]["redis"]["latency_ms"],
        "error": "ConnectionError",
    }
    assert "password" not in response.text  # seul le type d'exception est exposé


async def test_health_degraded_when_qdrant_times_out(client: AsyncClient, qdrant: AsyncMock) -> None:
    async def hang() -> None:
        await asyncio.sleep(1)  # > HEALTH_TIMEOUT_S = 0.05

    qdrant.get_collections.side_effect = hang

    response = await client.get("/health")

    assert response.status_code == 503
    qdrant_status = response.json()["services"]["qdrant"]
    assert qdrant_status["status"] == "down"
    assert qdrant_status["error"] == "TimeoutError"
    assert qdrant_status["latency_ms"] < 1000
