"""Vstupní tabulka smoketestu: párování plánů se zprávami jednoho jazyka."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import (
    InputNote,
    UnpairedNotesError,
    input_notes,
)

if TYPE_CHECKING:
    from mediparse.domain.note_plan import NotePlan
    from tests.conftest import PlannedNote


def _plan(planned_note: PlannedNote, note_id: str, *labels: Diagnosis) -> NotePlan:
    return planned_note.plan.model_copy(update={"note_id": note_id, "labels": labels})


def test_pairs_plans_with_texts_sorted_by_note_id(planned_note: PlannedNote) -> None:
    """Každá zpráva dostane subject_id a labely svého plánu; pořadí nezávisí na zdroji."""
    plans = (
        _plan(planned_note, "90000002-DS-1", Diagnosis.AKI),
        _plan(planned_note, "90000001-DS-1"),
    )
    texts = {"90000002-DS-1": "druhá", "90000001-DS-1": "první"}

    notes = input_notes(plans, texts, "en")

    subject = planned_note.plan.subject_id
    assert notes == (
        InputNote("90000001-DS-1", subject, "en", "první", frozenset()),
        InputNote("90000002-DS-1", subject, "en", "druhá", frozenset({Diagnosis.AKI})),
    )


def test_plan_without_text_is_refused(planned_note: PlannedNote) -> None:
    """Smoketest nad neúplným korpusem by nic nedokazoval."""
    plans = (_plan(planned_note, "90000001-DS-1"), _plan(planned_note, "90000002-DS-1"))

    with pytest.raises(
        UnpairedNotesError, match="Plány bez zprávy v jazyce en: 90000002-DS-1"
    ):
        input_notes(plans, {"90000001-DS-1": "první"}, "en")


def test_text_without_plan_is_refused(planned_note: PlannedNote) -> None:
    """Zpráva bez plánu nemá labely."""
    texts = {"90000001-DS-1": "první", "90000009-DS-1": "cizí"}

    with pytest.raises(
        UnpairedNotesError, match="Zprávy bez plánu v jazyce en: 90000009-DS-1"
    ):
        input_notes((_plan(planned_note, "90000001-DS-1"),), texts, "en")


def test_duplicate_plans_are_refused(planned_note: PlannedNote) -> None:
    """Dva plány jedné zprávy by labely určilo pořadí ve zdroji."""
    plans = (
        _plan(planned_note, "90000001-DS-1"),
        _plan(planned_note, "90000001-DS-1", Diagnosis.AKI),
    )

    with pytest.raises(UnpairedNotesError, match="Duplicitní plány: 90000001-DS-1"):
        input_notes(plans, {"90000001-DS-1": "první"}, "en")


def test_empty_corpus_is_refused() -> None:
    """Tabulka bez řádků by smoketest prošla bez dat."""
    with pytest.raises(
        UnpairedNotesError, match="Korpus nemá v jazyce en žádnou zprávu"
    ):
        input_notes((), {}, "en")
