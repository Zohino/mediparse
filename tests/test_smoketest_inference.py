"""Doména inference smoketestu: soulad s tréninkem a výběr ukázky."""

from __future__ import annotations

import pytest

from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.smoketest_inference import (
    ModelMismatchError,
    demo_examples,
    ensure_predictions_present,
    ensure_same_predictions,
)
from tests.model_support import prediction


def test_same_predictions_within_tolerance_pass() -> None:
    """Skóre odlišné o zlomek tolerance a stejný label projdou."""
    expected = [prediction("n1", truth=True, score=0.5)]

    ensure_same_predictions(
        expected, [Classification(predicted=True, score=0.5 + 1e-12)]
    )


def test_different_label_names_note_id() -> None:
    """Jiný label vyhodí chybu s note_id a oběma hodnotami."""
    expected = [prediction("n1", truth=True, score=0.5)]

    with pytest.raises(ModelMismatchError, match="n1") as raised:
        ensure_same_predictions(expected, [Classification(predicted=False, score=0.5)])

    assert isinstance(raised.value, InvalidInputError)


def test_score_outside_tolerance_names_note_id() -> None:
    """Skóre mimo toleranci vyhodí chybu s note_id a oběma skóre."""
    expected = [
        prediction("n1", truth=True, score=0.5),
        prediction("n2", truth=False, score=0.5),
    ]
    actual = [
        Classification(predicted=True, score=0.5),
        Classification(predicted=True, score=0.6),
    ]

    with pytest.raises(ModelMismatchError, match=r"n2.*0\.5.*0\.6"):
        ensure_same_predictions(expected, actual)


def test_different_length_raises() -> None:
    """Různý počet predikcí a klasifikací je nesoulad."""
    with pytest.raises(ModelMismatchError, match="počet"):
        ensure_same_predictions([prediction("n1", truth=True, score=0.5)], [])


def test_demo_examples_positive_before_negative_smallest_note_id() -> None:
    """Ukázka je nejmenší pozitivní a pak nejmenší negativní note_id."""
    items = [
        prediction("n3", truth=False, score=0.5),
        prediction("n2", truth=True, score=0.5),
        prediction("n1", truth=False, score=0.5),
        prediction("n4", truth=True, score=0.5),
    ]

    assert [item.note_id for item in demo_examples(items)] == ["n2", "n1"]


def test_demo_examples_missing_class_returns_one() -> None:
    """Chybí-li třída, vrátí se jen druhá."""
    items = [
        prediction("n2", truth=False, score=0.5),
        prediction("n1", truth=False, score=0.5),
    ]

    assert [item.note_id for item in demo_examples(items)] == ["n1"]


def test_demo_of_empty_predictions_is_empty() -> None:
    """Prázdný vstup dá prázdnou n-tici."""
    assert demo_examples([]) == ()


def test_empty_predictions_raise() -> None:
    """Prázdné predikce nejsou co ověřit."""
    with pytest.raises(InvalidInputError, match="není co ověřit"):
        ensure_predictions_present([])
