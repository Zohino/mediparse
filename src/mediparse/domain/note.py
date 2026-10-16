"""Zpráva a její identifikátor ve skladbě MIMIC-IV-Note: ``subject_id-DS-pořadí``."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mediparse.domain.identity import SubjectId

# Bez úvodní nuly a nuly: int() by "010" a "10" sloučil v jednoho pacienta.
_NOTE_ID: Final = re.compile(r"(?P<subject>[1-9]\d*)-DS-\d+")


def note_id(subject_id: SubjectId, order: int) -> str:
    """Note_id propouštěcí zprávy pacienta ve skladbě MIMIC-IV-Note.

    Returns:
        Identifikátor ``subject_id-DS-pořadí``.
    """
    return f"{subject_id}-DS-{order}"


class InvalidNoteIdError(ValueError):
    """Jméno zprávy není note_id propouštěcí zprávy ve skladbě MIMIC-IV-Note."""


def subject_of(note_id: str) -> SubjectId:
    """Subject_id pacienta z note_id propouštěcí zprávy ve skladbě MIMIC-IV-Note.

    Returns:
        Identifikátor pacienta.

    Raises:
        InvalidNoteIdError: Jméno neodpovídá skladbě ``subject_id-DS-pořadí``.
    """
    match = _NOTE_ID.fullmatch(note_id)
    if match is None:
        msg = f"Neplatné note_id propouštěcí zprávy: {note_id!r}"
        raise InvalidNoteIdError(msg)
    return int(match["subject"])
