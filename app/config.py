"""Application configuration loaded from the environment."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable application settings.

    All values are sourced from environment variables (with .env support). The settings
    object is frozen — anything that needs runtime configuration should read it once at
    startup and stash it locally rather than mutating this.
    """

    model_config = SettingsConfigDict(env_file=".env", frozen=True, extra="ignore")

    llm_provider: Literal["anthropic", "openai", "google"] = "google"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    log_level: str = "INFO"
    app_env: Literal["local", "dev", "prod"] = "local"

    # CP2: persistence wiring.
    database_url: str = "postgresql+asyncpg://drift:secret@localhost:5433/drift_triage"
    redis_url: str = "redis://localhost:6379/2"

    # CP3 carry-over: trace-replay tooling. Optional.
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "drift-triage-agent"
    langsmith_tracing: bool = False

    # CP3 carry-over: deterministic test mode.
    seed: int = 42
    deterministic_mode: bool = False
    test_deterministic: bool = False

    # Drift-triage: platform integration.
    platform_url: str = "http://platform:8000"
    platform_webhook_secret: SecretStr | None = None
    dashboard_callback_url: str | None = None

    # ----- CP4: queue layer ---------------------------------------------------
    # Same Redis instance as the checkpointer is fine; we just isolate the
    # keyspace via a different db number. Production deployments pointing both
    # at the same db won't break, but you lose the "FLUSHDB on the queue can't
    # wipe checkpoints" guarantee.
    queue_redis_url: str = "redis://localhost:6379/3"

    # Master switch. ``False`` means agent tools execute in-process — the
    # queue layer is not consulted at all. We keep it on by default in CP4
    # because the whole lesson is the queue path; production should always
    # have it on.
    enable_queue: bool = True

    # When True AND a worker is unreachable within ``worker_reach_deadline_s``,
    # tools fall back to in-process execution rather than failing the run.
    # Convenient for local dev, dangerous in production — graceful degradation
    # is a policy choice, not an obvious default.
    enable_queue_fallback: bool = False

    # Worker pool. arq runs ``worker_concurrency`` coroutines per process.
    # Horizontal scaling = more processes (compose --scale worker=N).
    worker_concurrency: int = 4

    # Per-job ceiling. The arq worker cancels the coroutine and marks the job
    # failed once this expires.
    job_timeout_s: int = 120

    # Retry policy. Total attempts = ``max_retries + 1``. Backoff is
    # exponential with jitter, capped at ``retry_backoff_max_s``.
    max_retries: int = 3
    retry_backoff_base_s: float = 2.0
    retry_backoff_max_s: float = 60.0

    # How long a producer waits to confirm a worker pool is reachable.
    worker_reach_deadline_s: float = 5.0

    # Optional webhook for DLQ alerting. When unset we still log a warning,
    # but no external notification fires.
    dlq_alert_webhook: str | None = None

    @property
    def is_deterministic(self) -> bool:
        """True if either deterministic switch is set."""
        return self.deterministic_mode or self.test_deterministic

    @property
    def alembic_database_url(self) -> str:
        """Sync-driver URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "")

    @model_validator(mode="after")
    def _check_provider_key(self) -> "Settings":
        """Reject configurations where the active provider has no API key."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    Tests should call ``get_settings.cache_clear()`` after mutating environment
    variables.
    """
    return Settings()
