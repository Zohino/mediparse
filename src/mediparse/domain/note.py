"""Zpráva a její identifikátor ve skladbě MIMIC-IV-Note: ``subject_id-DS-pořadí``."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mediparse.domain.identity import SubjectId

# Bez úvodní nuly a nuly: int() by "010" a "10" sloučil v jednoho pacienta.
_NOTE_ID: Final = re.compile(r"(?P<subject>[1-9]\d*)-DS-(?P<order>\d+)")


def note_id(subject_id: SubjectId, order: int) -> str:
    """Note_id propouštěcí zprávy pacienta ve skladbě MIMIC-IV-Note.

    Returns:
        Identifikátor ``subject_id-DS-pořadí``.
    """
    return f"{subject_id}-DS-{order}"


class NoteIdentityError(ValueError):
    """Společná báze chyb identity zprávy; text jejích potomků smí nést note_id."""


class InvalidNoteIdError(NoteIdentityError):
    """Jméno zprávy není note_id propouštěcí zprávy ve skladbě MIMIC-IV-Note."""


def _match_note_id(note_id: str) -> re.Match[str]:
    match = _NOTE_ID.fullmatch(note_id)
    if match is None:
        msg = f"Neplatné note_id propouštěcí zprávy: {note_id!r}"
        raise InvalidNoteIdError(msg)
    return match


def subject_of(note_id: str) -> SubjectId:
    """Subject_id pacienta z note_id propouštěcí zprávy ve skladbě MIMIC-IV-Note.

    Returns:
        Identifikátor pacienta.

    Raises:
        InvalidNoteIdError: Jméno neodpovídá skladbě ``subject_id-DS-pořadí``.
    """
    return int(_match_note_id(note_id)["subject"])


def order_of(note_id: str) -> int:
    """Pořadí propouštěcí zprávy v note_id ve skladbě MIMIC-IV-Note.

    Returns:
        Číselné pořadí zprávy.

    Raises:
        InvalidNoteIdError: Jméno neodpovídá skladbě ``subject_id-DS-pořadí``.
    """
    return int(_match_note_id(note_id)["order"])


class NoteSubjectMismatchError(NoteIdentityError):
    """Note_id zprávy patří jinému pacientovi, než kterého uvádí její řádek."""


def ensure_note_of_subject(note_id: str, subject_id: SubjectId) -> None:
    """Ověří, že note_id zprávy patří pacientovi ``subject_id``.

    Args:
        note_id: Identifikátor zprávy.
        subject_id: Pacient, kterého uvádí řádek zprávy.

    Raises:
        InvalidNoteIdError: Jméno neodpovídá skladbě ``subject_id-DS-pořadí``.
        NoteSubjectMismatchError: Note_id patří jinému pacientovi.
    """
    if subject_of(note_id) != subject_id:
        # Řádek nese reálné zprávy MIMIC; hláška ani log nesmí nést identifikátory.
        # Hláška InvalidNoteIdError ze subject_of note_id nese, chrání ji adaptér.
        msg = "note_id zprávy nepatří pacientovi uvedenému v subject_id."
        raise NoteSubjectMismatchError(msg)
