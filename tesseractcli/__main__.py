"""Entry point module.

`pyproject.toml` maps the `tesseract` console command to `main()` here:

    [project.scripts]
    tesseract = "tesseractcli.__main__:main"

So after `pip install -e .`, typing `tesseract` in any terminal runs this
file's `main()` function, which launches the Textual app — starting on the
WelcomeScreen (the ASCII logo).
"""

from tesseractcli.ui.app import run


def main() -> None:
    run()


if __name__ == "__main__":
    # Also allows `python -m tesseractcli` as an alternative to the
    # installed `tesseract` command, useful while developing.
    main()
