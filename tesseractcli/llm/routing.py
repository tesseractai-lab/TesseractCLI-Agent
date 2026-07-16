"""
tesseractcli/llm/routing.py
Domain model for "how do we pick a model for a given task" - owned
entirely by the LLM/agent layer. Deliberately independent from
Settings: Settings is flat, env-backed config; routing is structured
config with its own lifecycle (later: load/save to routing.json
without touching env parsing at all). The TUI settings screen and the
dispatcher both *consume* RoutingTable; neither owns it.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field


class RoutingStep (BaseModel):
    provider: str
    model: str


class RoutingConfig(BaseModel):
    primary: RoutingStep
    fallbacks: list[RoutingStep ] = []
    temperature: float = 0.3
    max_tokens: int = 4096


def default_main_model() -> RoutingConfig:
    """Fallback-of-last-resort routing, used when a task isn't present
    in RoutingTable.default_routing. Groq first (fast, generous free
    tier), Cerebras as backup - a starting point, not a measured
    choice; swap freely once real usage data exists."""
    return RoutingConfig(
        primary=RoutingStep (provider="groq", model="openai/gpt-oss-20b"),
        fallbacks=[RoutingStep (provider="cerebras", model="llama3.1-8b")],
    )


def default_task_routing() -> dict[str, RoutingConfig]:
    """Ships with a real per-task routing table out of the box - each
    task gets its own model + context sizing, matching the original
    design intent ("choose models with context sizing appropriate to
    each task"). These specific providers/models/token limits are a
    starting point - add, remove, or retune entries as real task types
    and provider preferences emerge; nothing else in the dispatcher
    depends on these particular choices."""
    return {
        "code_generation": RoutingConfig(
    primary=RoutingStep(provider="mistral", model="mistral-large-latest"),
    fallbacks=[
        RoutingStep(provider="cerebras", model="gpt-oss-120b"),
        RoutingStep(provider="huggingface", model="Qwen/Qwen2.5-72B-Instruct"),
        RoutingStep(provider="groq", model="llama-3.1-8b-instant"),
        RoutingStep(provider="together", model="Qwen2.5-Coder-32B-Instruct"),
    ],
    temperature=0.1,
    max_tokens=8192,
        ),
        "summarization": RoutingConfig(
            primary=RoutingStep(provider="cerebras", model="llama3.1-8b"),
            fallbacks=[
                RoutingStep(provider="groq", model="openai/gpt-oss-20b"),
                RoutingStep(provider="openrouter", model="meta-llama/llama-3.3-70b-instruct:free"),
            ],
            temperature=0.2,
            max_tokens=2048,
        ),
    }


class RoutingTable(BaseModel):
    """Everything the dispatcher needs to resolve a task name to a
    RoutingConfig. Construct directly for tests/overrides, or use
    get_routing_table() for the process-wide cached instance."""

    main_model: RoutingConfig = Field(default_factory=default_main_model)
    default_routing: dict[str, RoutingConfig] = Field(
        default_factory=default_task_routing
    )

    def resolve(self, task_name: str | None) -> RoutingConfig:
        """Task-specific routing if configured, else main_model."""
        if task_name and task_name in self.default_routing:
            return self.default_routing[task_name]
        return self.main_model


@lru_cache(maxsize=1)
def get_routing_table() -> RoutingTable:
    return RoutingTable()
