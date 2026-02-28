"""Database connection and session management for PostgreSQL.

Supports both async (for API/ingestion) and sync (for batch/ML) usage.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from lto.storage.models import Base

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "LTO_DATABASE_URL",
    "postgresql+asyncpg://lto:lto_password@localhost:5432/lto",
)

DATABASE_URL_SYNC = os.getenv(
    "LTO_DATABASE_URL_SYNC",
    "postgresql+psycopg2://lto:lto_password@localhost:5432/lto",
)

# ---------------------------------------------------------------------------
# Async engine (for FastAPI / API layer)
# ---------------------------------------------------------------------------

_async_engine = None
_async_session_factory = None


def _get_async_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _async_engine


def _get_async_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            _get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async database session."""
    factory = _get_async_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Sync engine (for batch ML training / scripts)
# ---------------------------------------------------------------------------

_sync_engine = None
_sync_session_factory = None


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            DATABASE_URL_SYNC,
            echo=False,
            pool_size=5,
        )
    return _sync_engine


def get_sync_session() -> Session:
    """Get a synchronous database session for batch operations."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(bind=_get_sync_engine())
    return _sync_session_factory()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = _get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def init_db_sync() -> None:
    """Create all tables synchronously (for scripts / CLI)."""
    engine = _get_sync_engine()
    Base.metadata.create_all(engine)
