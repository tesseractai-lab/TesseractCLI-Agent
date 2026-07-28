"""tesseractcli/models/config_models/agent_models.py"""

from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field

class AgentConfig(BaseModel):
    """Behavioral settings for the application's agent runtime.

    Attributes:
        temperature: Default sampling temperature for the agent when a
            pack-specific value is not applicable.
        max_iterations: Maximum number of reasoning/tool-call loops
            allowed per task before the agent must stop.
        timeout_seconds: Maximum wall-clock time, in seconds, allowed
            per task.
        lazy_tool_loading: If True, tools registered with `core=False`
            (see tools/registry.py) are withheld from the model until
            it calls `search_tools` for them, instead of always being
            sent - see agent/loop.py. Defaults to False: every
            registered tool's full schema is sent every request
            regardless of its `core` flag, which is the old/simple
            behavior and the right default while the tool count is
            small (the schema-size saving isn't worth an extra
            search_tools round trip yet). Flip to True once enough
            tools exist that most turns only touch a few of them.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_iterations: int = Field(default=10, ge=1)
    timeout_seconds: int = Field(default=120, ge=1)
    max_context_messages: int = Field(default=40, ge=1)
    lazy_tool_loading: bool = Field(default=False)
