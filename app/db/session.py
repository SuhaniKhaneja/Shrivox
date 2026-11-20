"""
Async engine + session factory, shared across the app via dependency injection.

The pool defaults are deliberately small for Neon/serverless Postgres and the
Render Free memory limit. `pool_pre_ping` detects connections dropped while a
free database was suspended; `pool_recycle` periodically replaces older ones.

Neon commonly supplies a libpq URL containing `sslmode=require` and sometimes
`channel_binding=require`. SQLAlchemy's asyncpg dialect passes connection
options to asyncpg, whose TLS switch is named `ssl`. Convert SSL mode into that
explicit connect argument and remove libpq-only query options before engine
creation. TLS remains mandatory when DATABASE_REQUIRE_TLS=true.
"""

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

_ALLOWED_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}


def database_url_and_options(config=settings):
    """Return an asyncpg URL and bounded engine options without opening a connection.

    `config` is injectable to keep the transformation unit-testable. A plain
    `postgres://` or `postgresql://` URL is upgraded to SQLAlchemy's asyncpg
    dialect. Neon/libpq `sslmode` is translated to asyncpg's `ssl` connect arg;
    `channel_binding` is removed because asyncpg does not expose that libpq URL
    option. Set DATABASE_REQUIRE_TLS=true in hosted environments to force TLS.
    """
    url = make_url(config.postgres_dsn)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+asyncpg")
    if url.drivername != "postgresql+asyncpg":
        raise ValueError("POSTGRES_DSN must use PostgreSQL; the application requires the asyncpg driver")

    query = dict(url.query)
    ssl_mode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    # Avoid passing URL-level SSL arguments and connect_args simultaneously.
    query_ssl = query.pop("ssl", None)
    url = url.set(query=query)

    if config.database_require_tls and ssl_mode == "disable":
        raise ValueError("DATABASE_REQUIRE_TLS=true conflicts with POSTGRES_DSN sslmode=disable")
    if config.database_require_tls:
        selected_ssl = ssl_mode if ssl_mode in {"verify-ca", "verify-full"} else "require"
    else:
        selected_ssl = ssl_mode or query_ssl
    if selected_ssl is not None and str(selected_ssl) not in _ALLOWED_SSL_MODES:
        raise ValueError(f"Unsupported PostgreSQL SSL mode: {selected_ssl}")

    connect_args = {"timeout": config.database_connect_timeout_seconds}
    if selected_ssl is not None:
        connect_args["ssl"] = str(selected_ssl)

    options = {
        "pool_size": config.database_pool_size,
        "max_overflow": config.database_max_overflow,
        "pool_timeout": config.database_pool_timeout_seconds,
        "pool_recycle": config.database_pool_recycle_seconds,
        "pool_pre_ping": True,
        "connect_args": connect_args,
    }
    return url, options


def create_database_engine(config=settings):
    url, options = database_url_and_options(config)
    return create_async_engine(url, echo=False, **options)


engine = create_database_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
