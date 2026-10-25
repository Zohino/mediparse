"""Text zprávy: jedna definice značky ___ pro počet i věkovou značku."""

from __future__ import annotations

from typing import Final

import pytest

from mediparse.domain.note_text import (
    AGE_MARKER,
    deid_markers,
    segment,
)

TWO: Final = 2


def test_marker_is_exactly_three_underscores() -> None:
    """Značka je právě ___, delší i kratší řady podtržítek se nepočítají."""
    assert deid_markers("x ___ y") == 1
    assert deid_markers("____") == 0
    assert deid_markers("__") == 0
    assert deid_markers("______") == 0
    assert deid_markers("___ ___") == TWO


def test_age_marker_needs_standalone_marker() -> None:
    """Věková značka je ___ year old, ne věk za delší řadou podtržítek."""
    assert AGE_MARKER.search("a ___ year old woman") is not None
    assert AGE_MARKER.search("a ____ year old woman") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("no headers", ("no headers", ()), id="bez hlavičky"),
        pytest.param("", ("", ()), id="prázdný text"),
        pytest.param(
            "see Allergies: none",
            ("see Allergies: none", ()),
            id="hlavička uprostřed řádku",
        ),
        pytest.param(
            "pre\nAllergies:\nnone\nChief Complaint:\nfatigue",
            ("pre\n", (("allergies", "\nnone\n"), ("complaint", "\nfatigue"))),
            id="preambule a dvě sekce",
        ),
    ],
)
def test_segment_splits_preamble_and_sections(
    text: str, expected: tuple[str, tuple[tuple[str, str], ...]]
) -> None:
    """Text bez hlavičky je celý preambulí, s hlavičkami se dělí na jejich začátcích."""
    headers = {"allergies": "Allergies", "complaint": "Chief Complaint"}

    assert segment(text, headers) == expected
