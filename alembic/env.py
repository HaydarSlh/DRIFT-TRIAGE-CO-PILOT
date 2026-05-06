"""Alembic environment.

Reads ``DATABASE_URL`` from the environment, strips the ``+asyncpg`` driver
suffix (Alembic's online runner is sync), and runs migrations against the
metadata declared in ``app.db.base.Base``.

Importing ``app.db.models`` is necessary even though the symbol isn't used —
the import side-effect is what registers the model classes against
``Base.metadata`` so autogenerate sees them.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db import models  # noqa: F401  -- side-effect import; registers tables
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the URL from settings into Alembic's config so this single source of
# truth is the .env file.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.alembic_database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout for inspection without connecting to a DB."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
