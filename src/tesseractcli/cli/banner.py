from rich.panel import Panel
from rich.text import Text

TESSERACT_LOGO = """\
"""

ASCII_TITLE = """\
 ████████╗███████╗███████╗███████╗███████╗██████╗  █████╗  ██████╗████████╗
 ╚══██╔══╝██╔════╝██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
    ██║   █████╗  ███████╗███████╗█████╗  ██████╔╝███████║██║        ██║
    ██║   ██╔══╝  ╚════██║╚════██║██╔══╝  ██╔══██╗██╔══██║██║        ██║
    ██║   ███████╗███████║███████║███████╗██║  ██║██║  ██║╚██████╗   ██║
    ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝"""

def print_banner():
    logo = Text(TESSERACT_LOGO, style="bold cyan")
    title = Text(ASCII_TITLE, style="cyan")
    tagline = Text("\n\n⚡ AI Agent Runtime • v0.1.0\t", style="dim")

    content = Text.assemble(logo, "\n", title, tagline)

    panel = Panel.fit(
        content,
        border_style="cyan",
        padding=(1, 15),
    )
    return panel
