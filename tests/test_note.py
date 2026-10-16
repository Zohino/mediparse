"""Identifikátor zprávy: note_id ve skladbě MIMIC-IV-Note a z něj subject_id."""

from typing import Final

import pytest

from mediparse.domain.note import InvalidNoteIdError, note_id, subject_of

MIMIC_SUBJECT: Final = 10000032


def test_subject_is_parsed_from_note_id() -> None:
    """Subject_id je první část note_id ve skladbě MIMIC-IV-Note, jako číslo."""
    subject = subject_of("10000032-DS-21")

    assert subject == MIMIC_SUBJECT
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
    subject = 90000123

    assert subject_of(note_id(subject, 4)) == subject
