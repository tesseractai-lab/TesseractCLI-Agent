# tesseractcli/memory/migrations/runner.py

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent
ALEMBIC_INI = MIGRATIONS_DIR / "alembic.ini"


def run_migrations(db_path: Path) -> None:
    """Upgrade the workspace database to the latest Alembic revision."""
    config = Config(str(ALEMBIC_INI))

    config.set_main_option(
        "script_location",
        str(MIGRATIONS_DIR),
    )
    config.set_main_option(
        "sqlalchemy.url",
        f"sqlite:///{db_path.resolve()}",
    )

    command.upgrade(config, "head")
