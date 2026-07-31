"""tesseractcli/models/config_models/global_config_models.py"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tesseractcli.models.config_models.agent_models import AgentConfig
from tesseractcli.models.config_models.path_models import PathsConfig
from tesseractcli.models.config_models.provider_models import ModelPack
from tesseractcli.models.config_models.verbose_models import VerboseConfig


class GlobalConfig(BaseModel):
    """Root model representing the full contents of ``global_config.yaml``.

    Attributes:
        schema_version: Version of the configuration schema. Used to
            detect and, in the future, migrate older configuration
            files.
        providers: Mapping of provider pack name (e.g. ``"main"``,
            ``"vision"``) to its :class:`ModelPack` definition. Packs
            are open-ended: arbitrary keys may be added without any
            change to this model.
        paths: Filesystem path configuration.
        agent: Agent runtime configuration.
        verbose: Verbosity toggles (e.g. `verbose.errors`) for how much
            internal detail gets printed to the scrollback.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="1.0", description="Configuration schema version."
    )
    providers: dict[str, ModelPack] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    verbose: VerboseConfig = Field(default_factory=VerboseConfig)
