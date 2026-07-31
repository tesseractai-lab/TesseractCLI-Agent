"""
tesseractcli.models.tools_result.py

The shared contract every tool in this package returns: `ToolResult`.
"""

from typing import Any, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SideEffect(TypedDict):
    """A single side effect entry. Structured, not free text, so an approval
    layer can iterate over these generically across any tool."""

    type: str  # e.g. "created_directory", "overwrote_existing_file"
    detail: str  # human-readable specifics, e.g. the path involved


class BaseMetadata(TypedDict, total=False):
    """Fields any tool MIGHT include. Never required, since not every tool
    call produces every kind of information (e.g. a call that fails fast
    with a validation error has no meaningful duration to report)."""

    side_effects: list[SideEffect]
    duration_ms: float

# ---------------------------------------------------------------------------
# Per-tool metadata schemas
# ---------------------------------------------------------------------------

class ReadFileMetadata(BaseMetadata, total=False):
    path: str
    total_lines: int
    truncated: bool
    encoding: str
    hint: str


class WriteFileMetadata(BaseMetadata, total=False):
    path: str
    bytes_written: int
    mode: str  # "overwrite" | "append"


class EditFileMetadata(BaseMetadata, total=False):
    path: str
    match_count: int
    chars_replaced: int


class RunCommandMetadata(BaseMetadata, total=False):
    command: list[str]
    exit_code: int
    timed_out: bool
    blocked_by_policy: bool
    blocked_reason: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: float


class ListDirectoryMetadata(BaseMetadata, total=False):
    path: str
    item_count: int
    duration_ms: float

# ---------------------------------------------------------------------------
# The ToolResult contract itself
# ---------------------------------------------------------------------------

ToolMetadata: TypeAlias = (
    RunCommandMetadata | ReadFileMetadata | WriteFileMetadata | EditFileMetadata | ListDirectoryMetadata
)

class ToolResult(BaseModel):
    tool_name: str = Field(...)
    success: bool = Field(...)
    output: str = ""
    error: str | None = None
    metadata: ToolMetadata | dict[str, Any] = Field(default_factory=dict)
