"""Doména evaluace: ověření, že pravda diagnózy obsahuje obě třídy."""

from __future__ import annotations

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import (
    Prediction,
    ensure_both_classes,
    ensure_unique_keys,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note import NoteSubjectMismatchError
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
    """Jiná zpráva, diagnóza nebo řádek matice tvoří jiný klíč."""
    ensure_unique_keys([
        prediction("1-DS-1", truth=True, score=1.0),
        prediction("1-DS-2", truth=True, score=1.0),
        prediction("1-DS-1", truth=True, score=1.0, diagnosis=Diagnosis.CKD),
        prediction("1-DS-1", truth=True, score=1.0, row_id="other"),
    ])


def test_duplicate_key_is_rejected_without_note_id() -> None:
    """Opakovaný (row_id, note_id, diagnosis) je InvalidInputError s řádkem bez note_id."""
    with pytest.raises(
        InvalidInputError, match="řádek row, diagnóza diabetes"
    ) as raised:
        ensure_unique_keys([
            prediction("90000001-DS-1", truth=True, score=1.0),
            prediction("90000001-DS-1", truth=True, score=1.0),
        ])

    assert "90000001" not in str(raised.value)


def test_same_note_in_two_folds_is_rejected() -> None:
    """Zpráva ve dvou foldech by se ve sloučené metrice započetla dvakrát."""
    with pytest.raises(InvalidInputError, match="Opakovaná predikce"):
        ensure_unique_keys([
            prediction("1-DS-1", truth=True, score=1.0, fold=0),
            prediction("1-DS-1", truth=True, score=1.0, fold=1),
        ])


def test_prediction_of_another_subjects_note_is_rejected() -> None:
    """Predikci s note_id jiného pacienta než subject_id nejde vytvořit."""
    with pytest.raises(NoteSubjectMismatchError):
        Prediction(
            "row", 0, "1-DS-1", 2, Diagnosis.CKD, truth=True, predicted=True, score=1.0
        )
