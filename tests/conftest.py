"""Sdílené fixtures testů: syntetický korpus, referenční soubory MIMIC-IV-Note a config tabulek."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.infrastructure.mimic_reference import MimicReference

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

type Rows = Sequence[tuple[str, str]]

COMMIT: Final = "c" * 40
HEADER: Final = (
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
)
RADIOLOGY_ROWS: Final = (("10000033", "an unrelated radiology report"),)
HOSP: Final = {
    "url": "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
    "sha256": "f" * 64,
}


@dataclass(frozen=True)
class AuditFiles:
    """Korpus, reference a config tabulek pro audit a bránu v dočasném adresáři testu."""

    root: Path

    @property
    def tables(self) -> Path:
        """Config tabulek MIMIC, do kterého se reference připínají."""
        return self.root / "mimic_tables.json"

    def corpus(self, notes: Mapping[str, str]) -> Path:
        """Zapíše zprávy korpusu do repozitáře.

        Returns:
            Kořen korpusu uvnitř repozitáře.
        """
        repository = self.root / "repo"
        (repository / ".git").mkdir(parents=True, exist_ok=True)
        root = repository / "resources" / "synthetic"
        for name, text in notes.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def reference(self, name: str, rows: Rows) -> Path:
        """Zapíše referenční soubor ve tvaru tabulky MIMIC-IV-Note.

        Returns:
            Cesta k souboru ``.csv.gz``.
        """
        path = self.root / name
        with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(
                [f"{subject}-DS-1", subject, "1", "DS", "1", "", "", text]
                for subject, text in rows
            )
        return path

    def pin(self, references: Sequence[Path]) -> Path:
        """Zapíše config, který připíná otisky referencí; tabulka hosp do reference nepatří.

        Returns:
            Cesta ke configu tabulek.
        """
        entries = [
            {
                "url": f"https://physionet.org/files/mimic-iv-note/2.2/note/{path.name}",
                "sha256": MimicReference(path).sha256(),
            }
            for path in references
        ]
        entries.append(HOSP)
        self.tables.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")
        return self.tables

    def pinned_references(self, rows: Rows) -> tuple[Path, Path]:
        """Discharge se zadanými řádky a neutrální radiology, obě připnuté v configu.

        Returns:
            Cesty k discharge a radiology.
        """
        references = (
            self.reference("discharge.csv.gz", rows),
            self.reference("radiology.csv.gz", RADIOLOGY_ROWS),
        )
        self.pin(references)
        return references

    def audit_argv(
        self, corpus: Path, references: Sequence[Path], report: Path
    ) -> list[str]:
        """Argumenty auditu korpusu proti referencím a tomuto configu.

        Returns:
            Argumenty vstupního bodu auditu.
        """
        return [
            "--corpus",
            str(corpus),
            *(f"--reference={path}" for path in references),
            "--tables",
            str(self.tables),
            "--report",
            str(report),
            "--commit",
            COMMIT,
        ]


@pytest.fixture
def audit_files(tmp_path: Path) -> AuditFiles:
    """Soubory pro audit a bránu v dočasném adresáři testu.

    Returns:
        Továrna na korpus, reference a config tabulek.
    """
    return AuditFiles(tmp_path)
