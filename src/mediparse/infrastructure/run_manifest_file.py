"""Run manifest uložený jako JSON soubor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.run_manifest import RunManifest


@dataclass(frozen=True)
class JsonRunManifestFile:
    """Soubor s manifestem; JSON s odsazením 2 a koncovým řádkem."""

    path: Path

    def write(self, manifest: RunManifest) -> None:
        """Zapíše manifest a založí chybějící nadřazený adresář."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = manifest.model_dump_json(indent=2)
        self.path.write_text(f"{text}\n", encoding="utf-8")
