"""Řádek vstupní tabulky: zpráva v jednom jazyce s labely pětice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.language import Language


@dataclass(frozen=True)
class InputNote:
    """Řádek vstupní tabulky: text zprávy v jednom jazyce a labely diagnóz pětice."""

    note_id: str
    subject_id: SubjectId
    language: Language
    text: str
    labels: frozenset[Diagnosis]
