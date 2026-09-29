"""Plány syntetických zpráv na disku: jeden plán na řádek ve formátu JSON Lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.note_plan import NotePlan

PLANS_PATH: Final = CORPUS_ROOT / "plans.jsonl"


@dataclass(frozen=True)
class PlansFile:
    """Soubor plánů v UTF-8 s konci řádků LF a koncovým novým řádkem, jak je chtějí hooky repa."""

    path: Path

    def save(self, plans: Sequence[NotePlan]) -> None:
        """Zapíše plány v zadaném pořadí, každý na jeden řádek."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(f"{plan.model_dump_json()}\n" for plan in plans)
        self.path.write_bytes(content.encode("utf-8"))
