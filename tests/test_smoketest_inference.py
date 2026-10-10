"""Doména inference smoketestu: shoda s tréninkem a výběr ukázky."""

from __future__ import annotations

import pytest

from mediparse.domain.evaluation import Classification, Prediction
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_inference import (
    ModelMismatchError,
    demo_examples,
    ensure_predictions_present,
    ensure_same_predictions,
)


def _prediction(note_id: str, *, truth: bool, score: float = 0.5) -> Prediction:
    return Prediction(
        "row",
        0,
        note_id,
        1,
        Diagnosis.DIABETES,
        truth=truth,
        predicted=score > 0,
        score=score,
    )


def test_same_predictions_within_tolerance_pass() -> None:
    """Skóre odlišné o zlomek tolerance a stejný label projdou."""
    expected = [_prediction("n1", truth=True, score=0.5)]

    ensure_same_predictions(
        expected, [Classification(predicted=True, score=0.5 + 1e-12)]
    )


def test_different_label_names_note_id() -> None:
    """Jiný label vyhodí chybu s note_id a oběma hodnotami."""
    expected = [_prediction("n1", truth=True, score=0.5)]

    with pytest.raises(ModelMismatchError, match="n1") as raised:
        ensure_same_predictions(expected, [Classification(predicted=False, score=0.5)])

    assert isinstance(raised.value, InvalidInputError)


def test_score_outside_tolerance_names_note_id() -> None:
    """Skóre mimo toleranci vyhodí chybu s note_id a oběma skóre."""
    expected = [
        _prediction("n1", truth=True, score=0.5),
        _prediction("n2", truth=False),
    ]
    actual = [
        Classification(predicted=True, score=0.5),
        Classification(predicted=True, score=0.6),
    ]

    with pytest.raises(ModelMismatchError, match=r"n2.*0\.5.*0\.6"):
        ensure_same_predictions(expected, actual)


def test_different_length_raises() -> None:
    """Různý počet predikcí a klasifikací je neshoda."""
    with pytest.raises(ModelMismatchError, match="počet"):
        ensure_same_predictions([_prediction("n1", truth=True)], [])


def test_demo_examples_positive_before_negative_smallest_note_id() -> None:
    """Ukázka je nejmenší pozitivní a pak nejmenší negativní note_id."""
    items = [
        _prediction("n3", truth=False),
        _prediction("n2", truth=True),
        _prediction("n1", truth=False),
        _prediction("n4", truth=True),
    ]

    assert [item.note_id for item in demo_examples(items)] == ["n2", "n1"]


def test_demo_examples_missing_class_returns_one() -> None:
    """Chybí-li třída, vrátí se jen druhá."""
    items = [_prediction("n2", truth=False), _prediction("n1", truth=False)]

    assert [item.note_id for item in demo_examples(items)] == ["n1"]


def test_demo_examples_empty() -> None:
    """Prázdný vstup dá prázdnou n-tici."""
    assert demo_examples([]) == ()


def test_empty_predictions_raise() -> None:
    """Prázdné predikce nejsou co ověřit."""
    with pytest.raises(InvalidInputError, match="není co ověřit"):
        ensure_predictions_present([])
