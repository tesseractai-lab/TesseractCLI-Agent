"""
tesseractcli/tools/registry.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from tesseractcli.models import ToolResult

@dataclass
class ToolRegistry:
    _tools: dict[str, tuple[type[BaseModel], Callable]] = field(default_factory=dict)

    def add(self, name: str, schema: type[BaseModel], fn: Callable) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = (schema, fn)

    def schemas(self) -> list[type[BaseModel]]:
        """ All the plans for that were tracked for the LLM Provider. """
        return [schema for schema, _ in self._tools.values()]

    def tool_specs(self) -> list[tuple[str, type[BaseModel]]]:
        """Name + schema pairs, for building tool definitions that keep
        the LLM-facing tool name in sync with the registry key."""
        return [(name, schema) for name, (schema, _) in self._tools.items()]

    def dispatch(self, name: str, raw_args: dict, workspace_root: Path) -> ToolResult:
        """ "It is called when the LLM returns a tool_use request."""
        if name not in self._tools:
            return ToolResult(tool_name=name, success=False, output="",
                                error=f"Unknown tool '{name}'.")

        schema, fn = self._tools[name]

        try:
            validated = schema(**raw_args)
        except ValidationError as e:
            return ToolResult(tool_name=name, success=False, output="",
                                error=f"Invalid arguments for '{name}': {e}")

        return fn(workspace_root=workspace_root, **validated.model_dump())

