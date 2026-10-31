"""Doména inference smoketestu: soulad s tréninkem a výběr ukázky."""

from __future__ import annotations

import pytest

from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_inference import (
    ModelMismatchError,
    ensure_predictions_present,
    ensure_same_predictions,
)
from tests.model_support import prediction


def test_same_predictions_within_tolerance_pass() -> None:
    """Skóre odlišné o zlomek tolerance a stejný label projdou."""
    expected = [prediction("1-DS-1", truth=True, score=0.5)]

    ensure_same_predictions(
        expected, [Classification(predicted=True, score=0.5 + 1e-12)]
    )


def test_different_label_reports_position_without_identifiers() -> None:
    """Neshoda druhé ze tří syntetických zpráv neprozradí identifikátor."""
    expected = [
        prediction("90000001-DS-1", truth=True, score=0.5),
        prediction("90000002-DS-1", truth=True, score=0.5),
        prediction("90000003-DS-1", truth=True, score=0.5),
    ]
    actual = [
        Classification(predicted=True, score=0.5),
        Classification(predicted=False, score=0.5),
        Classification(predicted=True, score=0.5),
    ]

    with pytest.raises(ModelMismatchError) as raised:
        ensure_same_predictions(expected, actual)

    message = str(raised.value)
    assert "2." in message
    assert "3" in message
    assert "9000000" not in message
    assert "DS-" not in message
    assert "True" in message
    assert "False" in message
    assert "0.5" in message
    assert isinstance(raised.value, InvalidInputError)


def test_score_outside_tolerance_reports_position() -> None:
    """Skóre mimo toleranci vyhodí chybu s pořadím a oběma skóre."""
    expected = [
        prediction("1-DS-1", truth=True, score=0.5),
        prediction("2-DS-1", truth=False, score=0.5),
    ]
    actual = [
        Classification(predicted=True, score=0.5),
        Classification(predicted=True, score=0.6),
    ]

    with pytest.raises(ModelMismatchError, match=r"2\. predikce z 2.*0\.5.*0\.6"):
        ensure_same_predictions(expected, actual)


def test_different_length_raises() -> None:
    """Různý počet predikcí a klasifikací je nesoulad."""
    with pytest.raises(ModelMismatchError, match="počet"):
        ensure_same_predictions([prediction("1-DS-1", truth=True, score=0.5)], [])


def test_empty_predictions_raise() -> None:
    """Prázdné predikce nejsou co ověřit."""
    with pytest.raises(InvalidInputError, match="není co ověřit"):
        ensure_predictions_present([])
