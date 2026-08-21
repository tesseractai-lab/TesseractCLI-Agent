from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

from tesseractcli.memory.engine import build_engine
from tesseractcli.memory.migrations.runner import run_migrations

_engine_registry: dict[str, AsyncEngine] = {}


def _registry_key(workspace_root: Path | str) -> str:
    # Resolve so "./ws" and "/abs/path/ws" (same dir) collide to one key.
    return str(Path(workspace_root).resolve())


def get_engine(workspace_root: Path | str, *, echo: bool = False) -> AsyncEngine:
    """
    Returns the cached engine for this workspace if one already exists,
    otherwise runs migrations, builds the engine, and caches it.
    """
    key = _registry_key(workspace_root)

    if key not in _engine_registry:
        db_path = Path(workspace_root) / ".tesseract" / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        run_migrations(db_path)

        _engine_registry[key] = build_engine(db_path, echo=echo)

    return _engine_registry[key]


async def dispose_engine(workspace_root: Path | str) -> None:
    """
    Disposes the engine's connection pool and removes it from the registry.
    A later get_engine() for the same workspace builds a fresh engine.
    """
    key = _registry_key(workspace_root)
    engine = _engine_registry.pop(key, None)

    if engine is not None:
        await engine.dispose()
