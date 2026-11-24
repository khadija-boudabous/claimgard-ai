from functools import lru_cache

from pydantic import AnyHttpUrl, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration chargée depuis l'environnement / .env, validée au démarrage."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Services internes
    qdrant_url: AnyHttpUrl
    qdrant_grpc_port: int = 6334  # API en gRPC (HTTP/2 + protobuf), ingestion en REST
    redis_url: RedisDsn

    # Fournisseurs LLM (un seul suffit)
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Observabilité
    langfuse_host: AnyHttpUrl
    langfuse_public_key: str
    langfuse_secret_key: SecretStr

    # Healthcheck
    health_timeout_s: float = 2.0

    # RAG — changer un modèle impose de ré-ingérer la collection
    qdrant_collection: str = "contracts"
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "french"
    embed_cache_dir: str = "/app/.cache/fastembed"
    chunk_max_words: int = 150  # ≈ 200 tokens en français
    prefetch_limit: int = 20  # candidats par branche (dense / bm25) avant fusion RRF


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # valeurs fournies par l'env
