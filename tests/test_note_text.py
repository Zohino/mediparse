"""Text zprávy: jedna definice značky ___ pro počet i věkovou značku."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.note_text import (
    AGE_MARKER,
    deid_markers,
    measure_narrative,
    segment,
)

if TYPE_CHECKING:
    from mediparse.domain.note_text import NarrativeMeasure
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote

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


def _measure(
    body: str, planned_note: PlannedNote, config: SamplerConfig
) -> NarrativeMeasure:
    key = next(iter(planned_note.plan.section_words))
    return measure_narrative(((key, body),), planned_note.plan, config.structure)


def test_narrative_counts_only_standalone_markers(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Řada ______ není značka narativu, stejně jako v kontrole překladu."""
    measure = _measure("a ______ b ___ c", planned_note, sampler_config)

    assert measure.marks == 1


def test_narrative_marks_are_never_negative(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Věk za ____ není věková značka, takže se od značek neodečítá."""
    measure = _measure("a ____ year old woman", planned_note, sampler_config)

    assert measure.marks == 0


def test_narrative_measure_has_named_fields(
    planned_note: PlannedNote, sampler_config: SamplerConfig
) -> None:
    """Výsledek nese slova a značky jako pojmenovaná pole."""
    body = "a ___ year old woman with ___"
    measure = _measure(body, planned_note, sampler_config)

    assert (measure.words, measure.marks) == (len(body.split()), 1)
