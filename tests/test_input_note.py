"""Řádek vstupní tabulky: note_id patří pacientovi, kterého řádek uvádí."""

from __future__ import annotations

import pytest

from mediparse.domain.input_note import InputNote, NoteSubjectMismatchError
from mediparse.domain.language import Language
from mediparse.domain.note import InvalidNoteIdError

SUBJECT = 90000001


def test_note_of_its_subject_passes() -> None:
    """Zpráva, jejíž note_id skládá subject_id řádku, projde."""
    note = InputNote(f"{SUBJECT}-DS-2", SUBJECT, Language.EN, "text", frozenset())

    assert note.subject_id == SUBJECT


def test_note_of_another_subject_is_rejected_without_identifiers() -> None:
    """Note_id jiného pacienta odmítne už konstrukce a hláška nenese žádné id."""
    with pytest.raises(NoteSubjectMismatchError) as caught:
        InputNote(f"{SUBJECT}-DS-2", SUBJECT + 1, Language.EN, "text", frozenset())

    assert (
        str(caught.value) == "note_id zprávy nepatří pacientovi uvedenému v subject_id."
    )


def test_note_id_outside_mimic_composition_is_rejected() -> None:
    """Note_id mimo skladbu subject_id-DS-pořadí odmítne konstrukce."""
    with pytest.raises(InvalidNoteIdError):
        InputNote("a", 1, Language.EN, "text", frozenset())
