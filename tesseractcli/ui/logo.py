"""
ASCII logo assets for the Tesseract welcome screen.

The raw ASCII art is kept as a plain constant (no markup) so it can be
reused elsewhere (plain terminal fallback, README, etc.). The Rich/Textual
markup version with a gradient is built from it at import time.
"""

from __future__ import annotations
from tesseractcli.config.settings import get_settings
version = get_settings().APP_VERSION

# Plain ASCII art, no color markup.
ASCII_TITLE = r"""
   ████████╗███████╗███████╗███████╗███████╗██████╗  █████╗  ██████╗████████╗
   ╚══██╔══╝██╔════╝██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
      ██║   █████╗  ███████╗███████╗█████╗  ██████╔╝███████║██║        ██║
      ██║   ██╔══╝  ╚════██║╚════██║██╔══╝  ██╔══██╗██╔══██║██║        ██║
      ██║   ███████╗███████║███████║███████╗██║  ██║██║  ██║╚██████╗   ██║
      ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
""".strip("\n")

# Normalize left indentation while preserving the ASCII shape.
_lines = ASCII_TITLE.splitlines()
_common_indent = min(
    len(line) - len(line.lstrip(" "))
    for line in _lines
    if line.strip()
)
ASCII_TITLE = "\n".join(line[_common_indent:] for line in _lines)

# Width of the widest line.
LOGO_MIN_WIDTH = max(len(line) for line in ASCII_TITLE.splitlines())

# _GRADIENT_COLORS = [
#     "#f04dff",
#     "#ccff3f",
#     "#33ff88",
#     "#4f9fff",
#     "#7c8bff",
#     "#a87cff",
# ]
_GRADIENT_COLORS = [
    "#4dd8ff",
    "#3fc6ff",
    "#33b4ff",
    "#4f9fff",
    "#7c8bff",
    "#a87cff",
]

FALLBACK_TITLE = "[bold]TESSERACT[/bold]"

TAGLINE = f"[dim]a terminal-native coding agent [#ccff3f] v{version} [#ccff3f][/dim]"


def build_gradient_logo() -> str:
    """Return the ASCII title as Rich markup, one gradient color per line."""
    return "\n".join(
        f"[{_GRADIENT_COLORS[i % len(_GRADIENT_COLORS)]}]{line}[/{_GRADIENT_COLORS[i % len(_GRADIENT_COLORS)]}]"
        for i, line in enumerate(ASCII_TITLE.splitlines())
    )


GRADIENT_LOGO = build_gradient_logo()
