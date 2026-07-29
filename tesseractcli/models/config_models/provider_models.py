"""tesseractcli/models/config_models/provider_models"""

from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """A single concrete model reference used inside a provider pack.

    Attributes:
        provider: Name of the model provider (e.g. ``"openai"``,
            ``"anthropic"``, ``"ollama"``).
        model: Identifier of the model as understood by the provider
            (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    provider: str = Field(..., min_length=1, description="Model provider name.")
    model: str = Field(
        ..., min_length=1, description="Provider-specific model identifier."
    )

    def matches(self, provider: str, model: str) -> bool:
        """Check whether this entry matches the given provider/model pair.
        Args:
            provider: Provider name to compare against.
            model: Model identifier to compare against.
        Returns:
            True if both ``provider`` and ``model`` match exactly.
        """
        return self.provider == provider and self.model == model


class ModelPack(BaseModel):
    """A named collection of models grouped together for a single purpose.

    A pack couples a primary pool of models with an optional fallback
    pool, together with generation parameters shared by every model
    used through the pack (e.g. the ``main`` pack, the ``vision`` pack).

    Attributes:
        pool: Primary, ordered list of models used for this pack.
        fallback: Models used, in order, when every model in ``pool``
            fails or is unavailable.
        max_tokens: Maximum number of tokens to generate per request.
        temperature: Sampling temperature applied to requests made
            through this pack.
    """

    model_config = ConfigDict(extra="forbid")

    pool: list[ModelConfig] = Field(default_factory=list)
    fallback: list[ModelConfig] = Field(default_factory=list)
    max_tokens: int = Field(
        default=4096, ge=1, description="Maximum tokens per request."
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature."
    )
