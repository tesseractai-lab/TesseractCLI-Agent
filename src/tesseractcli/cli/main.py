
from rich.console import Console
import os
# from tesseractcli.cli.app import TesseractApp
from tesseractcli.cli.banner import print_banner

console = Console()



def main():
    console.print(print_banner())
    cwd = os.getcwd()
    console.print("Working on:", cwd)
    console.print("[green]✔ System ready[/green]")

    # TesseractApp().run()


if __name__ == "__main__":
    main()







