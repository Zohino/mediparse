"""Řádek vstupní tabulky: zpráva v jednom jazyce s labely pětice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.note import subject_of

if TYPE_CHECKING:
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.language import Language


class NoteSubjectMismatchError(ValueError):
    """Note_id zprávy patří jinému pacientovi, než kterého řádek uvádí."""


@dataclass(frozen=True)
class InputNote:
    """Řádek vstupní tabulky: text zprávy v jednom jazyce a labely diagnóz pětice."""

    note_id: str
    subject_id: SubjectId
    language: Language
    text: str
    labels: frozenset[Diagnosis]

    def __post_init__(self) -> None:
        """Ověří, že note_id patří pacientovi řádku; split dělí zprávy podle subject_id.

        Raises:
            NoteSubjectMismatchError: Note_id patří jinému pacientovi než ``subject_id``.
        """
        if subject_of(self.note_id) != self.subject_id:
            # Řádek nese reálné zprávy MIMIC; hláška ani log nesmí nést identifikátory.
            msg = "note_id zprávy nepatří pacientovi uvedenému v subject_id."
            raise NoteSubjectMismatchError(msg)
