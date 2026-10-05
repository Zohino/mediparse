"""Report tréninku jako plochý JSON se seřazenými klíči."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.smoketest_training import TrainingReport


@dataclass(frozen=True)
class JsonReportFile:
    """Soubor s reportem tréninku."""

    path: Path

    def write(self, report: TrainingReport) -> None:
        """Zapíše report a nadřazený adresář založí."""
        fields = asdict(report)
        metrics = fields.pop("metrics")
        document = {**fields, **metrics, "diagnosis": report.diagnosis.value}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
