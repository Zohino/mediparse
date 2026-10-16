"""Řádek vstupní tabulky: zpráva v jednom jazyce s labely svého plánu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediparse.domain.diagnosis import Diagnosis


@dataclass(frozen=True)
class InputNote:
    """Řádek vstupní tabulky: text zprávy v jednom jazyce a labely z jejího plánu."""

    note_id: str
    subject_id: int
    language: str
    text: str
    labels: frozenset[Diagnosis]
