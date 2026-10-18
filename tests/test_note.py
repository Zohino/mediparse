"""Identifikátor zprávy: note_id ve skladbě MIMIC-IV-Note a z něj subject_id."""

from typing import Final

import pytest

from mediparse.domain.note import (
    InvalidNoteIdError,
    NoteIdentityError,
    NoteSubjectMismatchError,
    ensure_note_of_subject,
    note_id,
    subject_of,
)

SUBJECT: Final = 90000123


def test_subject_is_parsed_from_note_id() -> None:
    """Subject_id je první část note_id ve skladbě MIMIC-IV-Note, jako číslo."""
    subject = subject_of("90000123-DS-4")

    assert subject == SUBJECT
    assert isinstance(subject, int)


@pytest.mark.parametrize(
    "name", ["90000123", "abc-DS-1", "90000123-RR-1", "0-DS-1", "010-DS-1"]
)
def test_malformed_note_id_is_rejected(name: str) -> None:
    """Jméno mimo skladbu, nulový pacient i úvodní nula do korpusu nepatří."""
    with pytest.raises(InvalidNoteIdError, match="note_id"):
        subject_of(name)


def test_note_id_and_subject_of_are_inverse() -> None:
    """Pacient se z vytvořeného note_id vrátí beze změny."""
    assert subject_of(note_id(SUBJECT, 4)) == SUBJECT


def test_note_of_its_subject_passes() -> None:
    """Note_id, které skládá subject_id pacienta, projde."""
    ensure_note_of_subject(note_id(SUBJECT, 2), SUBJECT)


def test_note_of_another_subject_is_rejected_without_identifiers() -> None:
    """Note_id jiného pacienta se odmítne a hláška nenese žádné id."""
    with pytest.raises(NoteSubjectMismatchError) as caught:
        ensure_note_of_subject(note_id(SUBJECT, 2), SUBJECT + 1)

    assert (
        str(caught.value) == "note_id zprávy nepatří pacientovi uvedenému v subject_id."
    )


def test_note_id_outside_mimic_composition_is_rejected_by_invariant() -> None:
    """Note_id mimo skladbu subject_id-DS-pořadí invariant odmítne."""
    with pytest.raises(InvalidNoteIdError):
        ensure_note_of_subject("a", 1)


@pytest.mark.parametrize("error", [InvalidNoteIdError, NoteSubjectMismatchError])
def test_identity_errors_share_a_base(error: type[ValueError]) -> None:
    """Adaptéry chytají obě chyby identity zprávy přes jedinou společnou bázi."""
    assert issubclass(error, NoteIdentityError)
