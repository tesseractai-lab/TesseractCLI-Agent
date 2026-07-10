"""
tesseractcli\models\tools_result.py

The shared contract every tool in this package returns: `ToolResult`.


"""

from pydantic import BaseModel, Field
from typing import TypeAlias, TypedDict, NotRequired, List


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SideEffect(TypedDict):
    """A single side effect entry. Structured, not free text, so an approval
    layer can iterate over these generically across any tool."""
    type: str      # e.g. "created_directory", "overwrote_existing_file"
    detail: str    # human-readable specifics, e.g. the path involved


class BaseMetadata(TypedDict, total=False):
    """Fields any tool MIGHT include. Never required, since not every tool
    call produces every kind of information (e.g. a call that fails fast
    with a validation error has no meaningful duration to report)."""
    side_effects: List[SideEffect]
    duration_ms: float


# ---------------------------------------------------------------------------
# Per-tool metadata schemas
# ---------------------------------------------------------------------------

class ReadFileMetadata(BaseMetadata, total=False):
    path: str
    total_lines: int
    truncated: bool
    encoding: str
    hint: str  # e.g. "File has 3400 lines, showing first 2000. Use start_line/end_line."


class WriteFileMetadata(BaseMetadata, total=False):
    path: str
    bytes_written: int
    mode: str  # "overwrite" | "append"


class EditFileMetadata(BaseMetadata, total=False):
    path: str
    match_count: int          # how many times old_str matched (for error diagnostics)
    chars_replaced: int


class RunCommandMetadata(BaseMetadata, total=False):
    command: list[str]
    exit_code: int
    timed_out: bool
    blocked_by_policy: NotRequired[bool]
    blocked_reason: NotRequired[str]

# ---------------------------------------------------------------------------
# The ToolResult contract itself
# ---------------------------------------------------------------------------

ToolMetadata : TypeAlias =  ( RunCommandMetadata | ReadFileMetadata | WriteFileMetadata | EditFileMetadata)


class ToolResult(BaseModel):
    """Standard result returned by every tool.

    Encapsulates the execution outcome, primary output, optional error
    information, and structured tool-specific metadata in a consistent
    format shared across all tools.
    """
    tool_name: str = Field(...)
    success: bool = Field(...)
    output: str
    error: str | None = None
    metadata : ToolMetadata  = Field(default_factory=dict)

