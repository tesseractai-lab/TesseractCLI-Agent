"""
tesseractcli/ui/views/settings_view.py

Renders the settings-stage landing page. Two config layers are shown
side by side, and it matters not to blur them:

- `config/settings.py` (`get_settings()`): process/.env-backed app
  settings - provider API keys (shown as configured/not-configured
  only, never the value), app name/version, log level. Read-only here.
- `config/global_config/manager.py` (`ConfigManager`): the actual
  `global_config.yaml` file - packs, the models inside each pack,
  paths, agent defaults. This is what `settings_commands.py` reads
  *and* writes; this view is just its landing page.

Previously this view only rendered `get_settings()` and never touched
ConfigManager at all, so nothing shown here reflected the YAML file or
let you see/edit the packs inside it - that's what this fixes, plus
wrapping the whole thing in a bordered panel (`ui.views.box`) instead
of a loose block of text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesseractcli.config.settings import get_settings
from tesseractcli.ui.views.box import render_box
from tesseractcli.ui.views.settings_commands import HELP_TEXT

if TYPE_CHECKING:
    from tesseractcli.ui.app import TesseractApp

_PROVIDER_KEY_FIELDS = [
    ("Cerebras", "CEREBRAS_API_KEY"),
    ("Groq", "GROQ_API_KEY"),
    ("Mistral", "MISTRAL_API_KEY"),
    ("HuggingFace", "HUGGINGFACE_API_KEY"),
    ("Anthropic", "ANTHROPIC_API_KEY"),
    ("Together AI", "TOGETHER_API_KEY"),
    ("OpenAI", "OPENAI_API_KEY"),
    ("Cohere", "COHERE_API_KEY"),
    ("OpenRouter", "OPENROUTER_API_KEY"),
    ("GitHub Models", "GITHUB_MODELS_TOKEN"),
    ("Google", "GOOGLE_API_KEY"),
]


def render_settings(app: "TesseractApp") -> Any:
    settings = get_settings()
    manager = app.config_manager
    cfg = manager.config

    provider_lines = []
    for label, field_name in _PROVIDER_KEY_FIELDS:
        configured = bool(getattr(settings, field_name, None))
        mark = "[green]✓ configured[/green]" if configured else "[dim]— not set[/dim]"
        provider_lines.append(f"  {label:<14} {mark}")

    workspace = app.workspace_root or "[dim](not set)[/dim]"
    pack = app.selected_pack or "[dim](not set)[/dim]"

    pack_summaries = "\n".join(
        f"  {name}: {len(p.pool)} pool / {len(p.fallback)} fallback"
        for name, p in cfg.providers.items()
    ) or "  [dim](no packs configured)[/dim]"

    body = (
        f"[bold]App[/bold]              {settings.APP_NAME} v{settings.APP_VERSION} ({settings.ENV_MODE.value})\n"
        f"[bold]Workspace[/bold]        {workspace}\n"
        f"[bold]Active pack[/bold]      {pack}\n"
        f"[bold]Config file[/bold]      {manager.config_path}\n"
        f"[bold]Schema version[/bold]   {cfg.schema_version}\n\n"
        "[bold]API keys[/bold]\n" + "\n".join(provider_lines) + "\n\n"
        "[bold]Packs (from global_config.yaml)[/bold]\n" + pack_summaries + "\n\n"
        f"{HELP_TEXT}"
    )
    return render_box("Settings", body)
