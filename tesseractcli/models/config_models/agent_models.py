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
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_iterations: int = Field(default=10, ge=1)
    timeout_seconds: int = Field(default=120, ge=1)
