"""Řádek vstupní tabulky: note_id patří pacientovi, kterého řádek uvádí."""

from __future__ import annotations

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.domain.note import InvalidNoteIdError, NoteSubjectMismatchError

SUBJECT = 90000001


def test_has_label_identifies_the_positive_class_of_each_diagnosis() -> None:
    """Label CKD znamená pozitivní CKD a negativní DIABETES."""
    note = InputNote("1-DS-1", 1, Language.EN, "text", frozenset({Diagnosis.CKD}))

    assert note.has_label(Diagnosis.CKD)
    assert not note.has_label(Diagnosis.DIABETES)


def test_empty_labels_have_no_positive_diagnosis() -> None:
    """Prázdné labely jsou negativní pro každou diagnózu."""
    note = InputNote("1-DS-1", 1, Language.EN, "text", frozenset())

    assert not any(note.has_label(diagnosis) for diagnosis in Diagnosis)


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
