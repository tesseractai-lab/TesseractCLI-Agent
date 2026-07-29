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

Each section below gets its own color pair (a saturated header color
plus a lighter/"open" tint of the same hue for its body) instead of
one flat block of default-colored text - makes it easier to visually
jump to "API keys" vs "Packs" vs the command reference at a glance
instead of reading top to bottom every time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesseractcli.config.settings import get_settings
from tesseractcli.ui.views.box import render_box
from tesseractcli.ui.views.settings_commands import HELP_TEXT, pack_color

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

# (header color, lighter/"open" body tint) per section - same hue,
# header is the saturated version and body is the washed-out one, so
# a section reads as one color family rather than two unrelated colors.
_SECTION_COLORS: dict[str, tuple[str, str]] = {
    "overview": ("#89DCEB", "#CFEFF7"),  # Cyan
    "keys":     ("#A6E3A1", "#D8F5D4"),  # Green
    "packs":    ("#CBA6F7", "#E8D8FF"),  # Purple
    "help":     ("#F9E2AF", "#FCECC8"),  # Amber
}

# Grey used for the divider dashes on either side of each section's
# title (item 6) - deliberately not one of the per-section hues above,
# so the divider itself reads as neutral scaffolding and the colored
# title still stands out as the actual section identity.
_DIVIDER_COLOR = "#236f9b"
_BORDER_COLOR = "#7F849C"

def _section(title: str, key: str, content: str) -> str:
    header_color, body_color = _SECTION_COLORS[key]

    width = 50
    left = "────────"
    right = "─" * max(0, width - len(title) - len(left) - 2)

    heading = (
        f"[{_BORDER_COLOR}]╭{left}[/]"
        f"[bold {header_color}] {title} [/bold {header_color}]"
        f"[{_BORDER_COLOR}]{right}[/]"
    )

    return f"{heading}\n[{body_color}]{content}[/{body_color}]"

def render_settings(app: "TesseractApp") -> Any:
    settings = get_settings()
    manager = app.config_manager
    cfg = manager.config

    provider_lines = []
    for label, field_name in _PROVIDER_KEY_FIELDS:
        configured = bool(getattr(settings, field_name, None))
        mark = "[green]✓ configured[/green]" if configured else "[dim]— not set[/dim]"
        provider_lines.append(f"  {label:<14} {mark}")

    workspace = app.workspace_root or "(not set)"
    pack = app.selected_pack or "(not set)"

    pack_summaries = "\n\n".join(
        f" [bold] [{pack_color(i)}]{name}[/{pack_color(i)}][bold]:\
            \n    |- {len(p.pool)} pool \
            \n    |- {len(p.fallback)} fallback "
        for i, (name, p) in enumerate(cfg.providers.items())
    ) or "  (no packs configured)"

    overview = (
        f"App              {settings.APP_NAME} v{settings.APP_VERSION} ({settings.ENV_MODE.value})\n"
        f"Config Dir       {manager.config_path.parent}\n"
        f"Workspace        {workspace}\n"
        f"Active pack      {pack}\n"
    )

    body = "\n\n".join(
        [
            _section("Overview", "overview", overview),
            _section("API keys", "keys", "\n".join(provider_lines)),
            _section("Packs (from global_config.yaml)", "packs", pack_summaries),
            _section("Commands", "help", HELP_TEXT),
        ]
    )
    return render_box("Settings", body)
