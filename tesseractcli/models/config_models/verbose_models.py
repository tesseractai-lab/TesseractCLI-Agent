"""tesseractcli/models/config_models/verbose_models.py"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerboseConfig(BaseModel):
    """Toggles for how much detail the app prints for internal events.

    Grouped under one ``verbose`` section (rather than a single
    top-level ``errors.verbose`` flag) so related toggles - `debug`,
    `log`, or similar in the future - have an obvious place to live
    next to `errors` instead of each needing its own top-level section.

    Attributes:
        errors: When True, `_report_error` (ui/app.py) prints the full
            exception traceback. When False (the default), only the
            exception type and message are shown, and the traceback is
            suppressed.
    """

    model_config = ConfigDict(extra="forbid")

    errors: bool = Field(
        default=False,
        description="Print full tracebacks on agent/tool errors instead of just type + message.",
    )
