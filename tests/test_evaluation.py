"""Doména evaluace: ověření, že pravda diagnózy obsahuje obě třídy."""

from __future__ import annotations

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import (
    Prediction,
    ensure_both_classes,
    ensure_complete_folds,
    ensure_unique_keys,
)
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.note import NoteSubjectMismatchError, note_id
from mediparse.domain.split import Split
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


SPLIT = Split((frozenset({1, 2}), frozenset({3}), frozenset({4})))
NOTES = tuple(
    InputNote(note_id(subject, order), subject, Language.EN, "text", frozenset())
    for subject, order in [(1, 1), (1, 2), (2, 1), (3, 1), (4, 1)]
)
FOLD_0 = ("1-DS-1", "1-DS-2", "2-DS-1")


def _fold(
    notes: tuple[str, ...],
    fold: int = 0,
    row_id: str = "row",
    diagnosis: Diagnosis = Diagnosis.DIABETES,
) -> list[Prediction]:
    return [
        prediction(
            name, truth=True, score=1.0, fold=fold, row_id=row_id, diagnosis=diagnosis
        )
        for name in notes
    ]


def test_exact_coverage_of_folds_rows_and_diagnoses_passes() -> None:
    """Každá skupina (řádek, diagnóza, fold) pokrývá přesně test svého foldu."""
    predictions = [
        item
        for row_id in ("a", "b")
        for diagnosis in (Diagnosis.DIABETES, Diagnosis.CKD)
        for item in (
            *_fold(FOLD_0, 0, row_id, diagnosis),
            *_fold(("3-DS-1",), 1, row_id, diagnosis),
        )
    ]

    ensure_complete_folds(predictions, NOTES, SPLIT)


def test_fold_missing_from_predictions_is_not_checked() -> None:
    """Smoketest trénuje jediný fold; chybějící fold 1 a 2 není chyba."""
    ensure_complete_folds(_fold(FOLD_0), NOTES, SPLIT)


def test_missing_note_is_rejected_with_count_only() -> None:
    """Chybějící testovací zpráva je chyba s počtem a bez note_id."""
    with pytest.raises(
        InvalidInputError,
        match=r"řádek row, diagnóza diabetes, fold 0.*chybí 1, navíc 0",
    ) as raised:
        ensure_complete_folds(_fold(FOLD_0[:2]), NOTES, SPLIT)

    assert "DS" not in str(raised.value)


def test_note_of_another_fold_is_rejected_as_extra() -> None:
    """Zpráva testu jiného foldu je navíc."""
    with pytest.raises(InvalidInputError, match="chybí 0, navíc 1"):
        ensure_complete_folds(_fold((*FOLD_0, "3-DS-1")), NOTES, SPLIT)


def test_swapped_note_of_same_count_is_rejected() -> None:
    """Výměna jedné zprávy za cizí zachová počet, ale ne množinu."""
    with pytest.raises(InvalidInputError, match="chybí 1, navíc 1"):
        ensure_complete_folds(_fold(("1-DS-1", "1-DS-2", "3-DS-1")), NOTES, SPLIT)


def test_fold_out_of_range_is_rejected() -> None:
    """Fold mimo rozsah splitu odmítne holdout."""
    with pytest.raises(InvalidInputError, match="Fold 5 neexistuje"):
        ensure_complete_folds(_fold(FOLD_0, fold=5), NOTES, SPLIT)


def test_note_of_subject_outside_split_is_rejected() -> None:
    """Zpráva pacienta mimo split se odmítne už při vyhledání testu foldu."""
    stranger = InputNote("9-DS-1", 9, Language.EN, "text", frozenset())

    with pytest.raises(InvalidInputError, match="mimo split"):
        ensure_complete_folds(_fold(FOLD_0), (*NOTES, stranger), SPLIT)
