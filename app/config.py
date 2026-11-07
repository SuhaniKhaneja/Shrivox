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

from typing import Literal

from pydantic import Field, model_validator
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
    # Lifetime of a customer chat ticket. The ticket is only needed to open the
    # WebSocket, and it travels in the URL, so keep it short. Bounded so a
    # misconfiguration cannot turn it back into a long-lived credential.
    chat_ticket_ttl_seconds: int = Field(default=60, ge=5, le=900)

    # --- Deployment mode and abuse controls ---
    # "production" turns on strict startup checks (see the JWT_SECRET validator).
    environment: Literal["development", "production"] = "development"
    # Self-service business signup. Open by default so local development works;
    # set REGISTRATION_ENABLED=false once real tenants are provisioned.
    registration_enabled: bool = True
    # Fixed-window limits, counted in Redis. If Redis is down the limiter fails
    # OPEN (logged), because failing closed would lock every customer out.
    rate_limit_enabled: bool = True
    rate_limit_login_per_ip_per_minute: int = Field(default=10, ge=1, le=100000)
    rate_limit_login_per_email_per_15min: int = Field(default=10, ge=1, le=100000)
    rate_limit_register_per_ip_per_hour: int = Field(default=5, ge=1, le=100000)
    rate_limit_ticket_per_user_per_minute: int = Field(default=30, ge=1, le=100000)
    rate_limit_messages_per_tenant_per_minute: int = Field(default=60, ge=1, le=100000)

    # --- Failure policy (see README "Failure handling") ---
    # Every call to Gemini (chat and embeddings) shares one policy: a per-attempt
    # timeout, and a bounded number of attempts for TRANSIENT errors only
    # (timeouts, 429, 5xx, network). Client errors (4xx) are never retried.
    upstream_timeout_seconds: float = Field(default=15.0, ge=1, le=120)
    upstream_max_attempts: int = Field(default=3, ge=1, le=5)
    upstream_retry_backoff_seconds: float = Field(default=0.5, ge=0, le=10)
    # Hard ceiling for one chat turn (intent + retrieval + generation + any
    # safety retry), so stacked retries can never leave a customer waiting minutes.
    turn_timeout_seconds: float = Field(default=45.0, ge=5, le=300)
    # Input limits. Oversized input is rejected before it costs an LLM call.
    max_message_chars: int = Field(default=2000, ge=1, le=20000)
    max_document_chars: int = Field(default=200_000, ge=1000, le=5_000_000)
    # Redis must fail fast: a hung Redis would otherwise hang every chat turn.
    redis_timeout_seconds: float = Field(default=2.0, ge=0.1, le=30)

    # --- App behavior ---
    # Messages kept per session (a customer/agent turn is TWO messages). Bounded:
    # 0 would make Redis' LTRIM keep everything, i.e. silently disable the cap.
    max_conversation_turns: int = Field(default=30, ge=2, le=500)
    session_ttl_seconds: int = Field(default=3600, ge=60)  # Redis session expiry (1 hr of inactivity)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_jwt_secret(self):
        """The JWT secret signs every credential, so a guessable one is a full
        authentication bypass. Production refuses to start with a weak one;
        development stays permissive (CI and local runs use short throwaway
        secrets). An empty secret is never acceptable."""
        secret = self.jwt_secret
        if not secret.strip():
            raise ValueError("JWT_SECRET must not be empty")
        if self.environment == "production":
            if len(secret) < 32:
                raise ValueError("JWT_SECRET must be at least 32 characters when ENVIRONMENT=production")
            if any(marker in secret.lower() for marker in _PLACEHOLDER_MARKERS):
                raise ValueError("JWT_SECRET looks like a placeholder; generate a random one (e.g. `openssl rand -hex 32`)")
            if len(set(secret)) < 8:
                raise ValueError("JWT_SECRET has too little variety to be random")
        return self


# Substrings that mark a copied-from-the-docs secret rather than a generated one.
_PLACEHOLDER_MARKERS = ("change_me", "changeme", "change-me", "your_secret", "your-secret", "replace_me", "example")

settings = Settings()
