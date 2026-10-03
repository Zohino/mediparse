"""Plány syntetických zpráv na disku: jeden plán na řádek ve formátu JSON Lines a z nich odvozené labely."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.labels import Diagnosis
from mediparse.domain.note_plan import NotePlan
from mediparse.infrastructure.input_file import parse_file
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

PLANS_PATH: Final = CORPUS_ROOT / "plans.jsonl"
LABELS_PATH: Final = CORPUS_ROOT / "labels.csv"


@dataclass(frozen=True)
class PlansFile:
    """Soubor plánů v UTF-8 s konci řádků LF a koncovým novým řádkem, jak je chtějí hooky repa."""

    path: Path

    def save(self, plans: Sequence[NotePlan]) -> None:
        """Zapíše plány v zadaném pořadí, každý na jeden řádek."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(f"{plan.model_dump_json()}\n" for plan in plans)
        self.path.write_bytes(content.encode("utf-8"))

    def load(self) -> tuple[NotePlan, ...]:
        """Načte plány v pořadí souboru.

        Returns:
            Plány zpráv.
        """
        return parse_file(self.path, _plans)


@dataclass(frozen=True)
class LabelsFile:
    """Labely zpráv v CSV: note_id, subject_id a 0/1 pro každou diagnózu v pořadí diagnóz."""

    path: Path

    def save(self, plans: Sequence[NotePlan]) -> None:
        """Zapíše labely plánů v UTF-8 s konci řádků LF, jak je chtějí hooky repa."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("note_id", "subject_id", *Diagnosis))
            writer.writerows(_label_row(plan) for plan in plans)


@dataclass(frozen=True)
class SyntheticPlanFiles:
    """Plány a z nich odvozené labely: ``plans.jsonl`` a ``labels.csv`` zapsané spolu."""

    plans: Path
    labels: Path

    def save(self, plans: Sequence[NotePlan]) -> None:
        """Zapíše plány a jejich labely."""
        PlansFile(self.plans).save(plans)
        LabelsFile(self.labels).save(plans)


def _label_row(plan: NotePlan) -> tuple[str | int, ...]:
    return (plan.note_id, plan.subject_id, *(int(d in plan.labels) for d in Diagnosis))


def _plans(content: str) -> tuple[NotePlan, ...]:
    return tuple(NotePlan.model_validate_json(line) for line in content.splitlines())
