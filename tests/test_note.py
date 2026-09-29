"""Identifikátor zprávy: note_id ve skladbě MIMIC-IV-Note a z něj subject_id."""

import pytest

from mediparse.domain.note import InvalidNoteIdError, subject_of


def test_subject_is_parsed_from_note_id() -> None:
    """Subject_id je první část note_id ve skladbě MIMIC-IV-Note."""
    assert subject_of("90000123-DS-4") == "90000123"


@pytest.mark.parametrize("note_id", ["90000123", "abc-DS-1", "90000123-RR-1"])
def test_malformed_note_id_is_rejected(note_id: str) -> None:
    """Soubor, jehož jméno není note_id propouštěcí zprávy, do korpusu nepatří."""
    with pytest.raises(InvalidNoteIdError, match="note_id"):
        subject_of(note_id)
