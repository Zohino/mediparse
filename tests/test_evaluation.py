"""Doména evaluace: ověření, že pravda diagnózy obsahuje obě třídy."""

from __future__ import annotations

import pytest

from mediparse.domain.evaluation import (
    Prediction,
    ensure_both_classes,
    ensure_unique_keys,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis


def _prediction(note: str, fold: int) -> Prediction:
    return Prediction(
        "row",
        fold,
        note,
        1,
        Diagnosis.CKD,
        truth=True,
        predicted=True,
        score=1.0,
    )


def test_mixed_truth_passes() -> None:
    """Smíšená pravda projde."""
    ensure_both_classes(Diagnosis.CKD, [True, False, True])


@pytest.mark.parametrize("truth", [[True, True], [False, False, False]])
def test_single_class_is_rejected(truth: list[bool]) -> None:
    """Samé True i samé False skončí InvalidInputError s diagnózou."""
    with pytest.raises(InvalidInputError, match="ckd"):
        ensure_both_classes(Diagnosis.CKD, truth)


def test_unique_keys_pass() -> None:
    """Různé klíče projdou."""
    ensure_unique_keys([
        _prediction("n1", 0),
        _prediction("n1", 1),
        _prediction("n2", 0),
    ])


def test_duplicate_key_is_rejected() -> None:
    """Opakovaný (row_id, fold, note_id, diagnosis) je InvalidInputError s klíčem."""
    with pytest.raises(InvalidInputError, match="n1"):
        ensure_unique_keys([_prediction("n1", 0), _prediction("n1", 0)])
