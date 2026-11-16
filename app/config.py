from functools import lru_cache

from pydantic import AnyHttpUrl, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration chargée depuis l'environnement / .env, validée au démarrage."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Services internes
    qdrant_url: AnyHttpUrl
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


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # valeurs fournies par l'env
