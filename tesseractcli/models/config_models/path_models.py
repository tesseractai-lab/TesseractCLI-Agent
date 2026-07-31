"""tesseractcli/models/config_models/path_models.py"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    """Filesystem locations used throughout the application.

    All paths may be relative (interpreted relative to the process's
    working directory or an application-defined root) or absolute.

    Attributes:
        data_dir: Directory used to store application data.
        cache_dir: Directory used for cached artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Field(
        default=Path("data"), description="Application data directory."
    )
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory.")
