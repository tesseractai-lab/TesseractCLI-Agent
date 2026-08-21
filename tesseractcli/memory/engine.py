from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine


def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """
    Fires on every new DBAPI connection (event: "connect") — SQLite PRAGMAs
    are connection-scoped, not database-scoped, so this must re-run per
    connection, not once at engine-creation time.
    """
    cursor = dbapi_connection.cursor()

    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")

    cursor.close()


def build_engine(db_path: Path | str, *, echo: bool = False) -> AsyncEngine:
    """
    One engine per workspace's SQLite file (e.g. `<workspace>/.tesseract/memory.db`).
    Not a process-wide singleton — each workspace is isolated, mirroring the
    per-workspace memory design.

    `db_path=":memory:"` also works, for tests.
    """
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=echo,
    )

    event.listens_for(engine.sync_engine, "connect")(_set_sqlite_pragma)

    return engine


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
