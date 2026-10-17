"""Doména evaluace: ověření, že pravda diagnózy obsahuje obě třídy."""

from __future__ import annotations

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import ensure_both_classes, ensure_unique_keys
from mediparse.domain.inputs import InvalidInputError
from tests.model_support import prediction


def test_mixed_truth_passes() -> None:
    """Smíšená pravda projde."""
    ensure_both_classes(Diagnosis.CKD, [True, False, True])


@pytest.mark.parametrize("truth", [[True, True], [False, False, False], []])
def test_single_class_is_rejected(truth: list[bool]) -> None:
    """Samé True, samé False i prázdná pravda skončí InvalidInputError s diagnózou."""
    with pytest.raises(InvalidInputError, match="ckd"):
        ensure_both_classes(Diagnosis.CKD, truth)


def test_unique_keys_pass() -> None:
    """Různé klíče projdou."""
    ensure_unique_keys([
        prediction("n1", truth=True, score=1.0, fold=0),
        prediction("n1", truth=True, score=1.0, fold=1),
        prediction("n2", truth=True, score=1.0, fold=0),
    ])


def test_duplicate_key_is_rejected() -> None:
    """Opakovaný (row_id, fold, note_id, diagnosis) je InvalidInputError s klíčem."""
    with pytest.raises(InvalidInputError, match="n1"):
        ensure_unique_keys([
            prediction("n1", truth=True, score=1.0, fold=0),
            prediction("n1", truth=True, score=1.0, fold=0),
        ])
