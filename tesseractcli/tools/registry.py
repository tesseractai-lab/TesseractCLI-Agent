"""
tesseractcli/tools/registry.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from tesseractcli.models import ToolResult
from tesseractcli.observability import traced_tool_call

@dataclass
class ToolRegistry:
    _tools: dict[str, tuple[type[BaseModel], Callable, bool]] = field(default_factory=dict)

    def add(self, name: str, schema: type[BaseModel], fn: Callable, needs_approval: bool = True) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = (schema, fn, needs_approval)

    def schemas(self) -> list[type[BaseModel]]:
        """ All the plans for that were tracked for the LLM Provider. """
        return [schema for schema, _, _ in self._tools.values()]

    def tool_specs(self) -> list[tuple[str, type[BaseModel]]]:
        """Name + schema pairs, for building tool definitions that keep
        the LLM-facing tool name in sync with the registry key."""
        return [(name, schema) for name, (schema, _, _) in self._tools.items()]

    def needs_approval(self, name: str) -> bool:
        """Used by the approval flow to decide whether to prompt the user.
        Unknown tool names fail safe (treated as needing approval)."""
        if name not in self._tools:
            return True
        _, _, needs = self._tools[name]
        return needs

    def dispatch(self, name: str, raw_args: dict, workspace_root: Path) -> ToolResult:
        """ "It is called when the LLM returns a tool_use request."""
        if name not in self._tools:
            return ToolResult(tool_name=name, success=False, output="",
                                error=f"Unknown tool '{name}'.")

        schema, fn, _ = self._tools[name]

        try:
            validated = schema(**raw_args)
        except ValidationError as e:
            return ToolResult(tool_name=name, success=False, output="",
                                error=f"Invalid arguments for '{name}': {e}")

        # Plain Python call, not a LangChain Runnable, so it is NOT
        # auto-traced the way model.ainvoke() calls are - traced_tool_call
        # wraps it in a LangSmith run named after the real tool (e.g.
        # "edit_file"), and is a no-op when tracing is off.
        with traced_tool_call(name, raw_args):
            return fn(workspace_root=workspace_root, **validated.model_dump())
