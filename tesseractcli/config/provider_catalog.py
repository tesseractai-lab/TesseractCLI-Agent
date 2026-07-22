"""
tesseractcli/config/provider_catalog.py

Static suggestion list used only by the settings UI's "add pack" /
"add model" flows, so someone adding a provider they don't use every
day has example model IDs to start from instead of typing a raw
`provider model` string blind.

This is UI sugar only: it is not validated against any live provider
API, not read by the LLM dispatch path, and not authoritative - model
names/IDs drift over time, so treat these as a starting point to
confirm against the provider's own docs, not a guarantee the model
still exists or is still free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSuggestion:
    provider: str
    label: str
    example_models: tuple[str, ...]


PROVIDER_CATALOG: tuple[ProviderSuggestion, ...] = (
    ProviderSuggestion("groq", "Groq", ("llama-3.3-70b-versatile", "qwen/qwen3-32b", "gemma2-9b-it")),
    ProviderSuggestion("cerebras", "Cerebras", ("llama3.1-8b", "llama-3.3-70b")),
    ProviderSuggestion("mistral", "Mistral", ("mistral-small-latest", "open-mistral-nemo")),
    ProviderSuggestion("huggingface", "HuggingFace", ("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct")),
    ProviderSuggestion("anthropic", "Anthropic", ("claude-sonnet-4-6", "claude-haiku-4-5")),
    ProviderSuggestion("together", "Together AI", ("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",)),
    ProviderSuggestion("openai", "OpenAI", ("gpt-4o-mini", "gpt-4o")),
    ProviderSuggestion("cohere", "Cohere", ("command-r7b", "command-r-plus")),
    ProviderSuggestion("openrouter", "OpenRouter", ("meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen-2.5-7b-instruct:free")),
    ProviderSuggestion("github_models", "GitHub Models", ("gpt-4o-mini", "Meta-Llama-3.1-8B-Instruct")),
    ProviderSuggestion("local_gguf", "Local GGUF", ("<path-to-local .gguf file>",)),
)


def suggestions_text() -> str:
    """Rich-markup block listing every provider + a few example models."""
    lines = []
    for entry in PROVIDER_CATALOG:
        models = ", ".join(entry.example_models)
        lines.append(f"  [bold]{entry.provider:<14}[/bold] {entry.label} — e.g. {models}")
    lines.append("")
    lines.append("[dim]Suggestions only - confirm current model IDs/pricing with the provider before relying on them.[/dim]")
    return "\n".join(lines)


def known_providers() -> list[str]:
    """Provider keys in the catalog, for validating/auto-completing `add model` input."""
    return [entry.provider for entry in PROVIDER_CATALOG]
