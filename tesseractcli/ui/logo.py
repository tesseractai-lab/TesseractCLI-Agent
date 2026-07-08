"""ASCII logo assets for the Tesseract welcome screen.

The raw ASCII art is kept as a plain constant (no markup) so it can be
reused elsewhere (plain terminal fallback, README, etc.). The Rich/Textual
markup version with a gradient is built from it at import time.
"""

from __future__ import annotations

# Plain ASCII art, no color markup — width is 78 chars, safe for any
# terminal >= 80 columns. Keep this as the single source of truth for the
# shape; regenerate via patorjk.com/software/taag ("ANSI Shadow" font) or
# `pyfiglet` if you ever want to change the wordmark.
ASCII_TITLE = r"""
 ████████╗███████╗███████╗███████╗███████╗██████╗  █████╗  ██████╗████████╗
 ╚══██╔══╝██╔════╝██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
    ██║   █████╗  ███████╗███████╗█████╗  ██████╔╝███████║██║        ██║
    ██║   ██╔══╝  ╚════██║╚════██║██╔══╝  ██╔══██╗██╔══██║██║        ██║
    ██║   ███████╗███████║███████║███████╗██║  ██║██║  ██║╚██████╗   ██║
    ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
""".strip("\n")

# Width of the widest line in ASCII_TITLE — used to decide whether the
# terminal is wide enough to render it, or whether to fall back to plain text.
LOGO_MIN_WIDTH = max(len(line) for line in ASCII_TITLE.split("\n"))

# A short vertical color gradient, top line to bottom line. Feel free to
# swap these for your own palette — any Rich-recognized color name or hex
# code works (e.g. "cyan", "#7c3aed").
_GRADIENT_COLORS = [
    "#4dd8ff",
    "#3fc6ff",
    "#33b4ff",
    "#4f9fff",
    "#7c8bff",
    "#a87cff",
]

# Plain fallback shown when the terminal is narrower than LOGO_MIN_WIDTH.
FALLBACK_TITLE = "[bold]TESSERACT[/bold]"

TAGLINE = "[dim]a terminal-native coding agent[/dim]"


def build_gradient_logo() -> str:
    """Return the ASCII title as Rich markup, one gradient color per line."""
    lines = ASCII_TITLE.split("\n")
    colors = _GRADIENT_COLORS
    styled = [
        f"[{colors[i % len(colors)]}]{line}[/{colors[i % len(colors)]}]"
        for i, line in enumerate(lines)
    ]
    return "\n".join(styled)


GRADIENT_LOGO = build_gradient_logo()
