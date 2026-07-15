"""
tesseractcli/models/tool_args.py

Pydantic models for tool arguments coming from the LLM.
This is the trust boundary: every tool receives raw, untrusted
input from the model's tool_use response, and these models validate
it before it reaches the actual tool function (see registry.dispatch()).
"""

from pydantic import BaseModel, Field


class WriteFileArgs(BaseModel):
    path: str = Field(description="Path to the file, relative to workspace root")
    content: str = Field(description="Content to write to the file")
    mode: str = Field(default="overwrite", description="'overwrite' or 'append'")


class ReadFileArgs(BaseModel):
    path: str = Field(description="Path to the file, relative to workspace root")
    start_line: int | None = Field(default=None, description="1-indexed start line, inclusive")
    end_line: int | None = Field(default=None, description="1-indexed end line, inclusive")


class EditFileArgs(BaseModel):
    path: str = Field(description="Path to the file, relative to workspace root")
    old_str: str = Field(description="Exact text to find; must be unique in the file")
    new_str: str = Field(description="Replacement text")


class ListDirectoryArgs(BaseModel):
    path: str = Field(default=".", description="Directory path, relative to workspace root")


class RunCommandArgs(BaseModel):
    command: list[str] = Field(description= """Argv list for the command, e.g. ['cat', 'path/to/file'].
        Do NOT wrap in a shell interpreter (no 'bash -c', 'sh -c',
        '-lc', etc.) — pass the target program and its arguments
        directly as separate list items.""")
