"""
Central configuration for Shrivox AI.

Why this exists:
Every service (Google, Postgres, Redis) needs credentials
and connection info. Scattering `os.getenv()` calls across the codebase
makes it impossible to know what config the app needs just by reading
one file, and it means typos in env var names fail silently at the
call site instead of at startup. Centralizing in one Settings object
means: (1) the app crashes at boot if required config is missing,
not mid-request, (2) every module imports the same validated object,
(3) tests can override settings cleanly via env vars or fixtures.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (reasoning) + Embeddings -- both Gemini, one provider, one free-tier budget ---
    google_api_key: str
    reasoning_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536

    # --- Postgres (also hosts pgvector -- see app/db/models.py for why) ---
    postgres_dsn: str = "postgresql+asyncpg://shrivox:shrivox@localhost:5432/shrivox"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Auth ---
    jwt_secret: str

    # --- App behavior ---
    max_conversation_turns: int = 30
    session_ttl_seconds: int = 3600  # Redis session expiry (1 hr of inactivity)
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
