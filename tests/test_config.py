import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings
from tests.conftest import REQUIRED_ENV


def _load() -> Settings:
    return Settings(_env_file=None)  # ignore le .env local : seul l'env du test compte


def test_settings_load_from_env() -> None:
    settings = _load()
    assert str(settings.qdrant_url).startswith("http://qdrant.test")
    assert settings.health_timeout_s == 0.05


def test_secrets_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    settings = _load()

    assert isinstance(settings.langfuse_secret_key, SecretStr)
    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.langfuse_secret_key.get_secret_value() == "sk-lf-super-secret"
    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert "sk-lf-super-secret" not in rendered
        assert "sk-openai-secret" not in rendered


@pytest.mark.parametrize("missing", sorted(REQUIRED_ENV))
def test_missing_required_variable_fails_fast(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.delenv(missing)
    with pytest.raises(ValidationError) as exc_info:
        _load()
    assert missing.lower() in {str(err["loc"][0]) for err in exc_info.value.errors()}


def test_invalid_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "not-a-url")
    with pytest.raises(ValidationError):
        _load()
