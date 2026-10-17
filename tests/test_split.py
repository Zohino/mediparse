"""Pravidla splitu: pacienti po foldech, třídy stratifikace a dost pacientů ve třídě."""

from __future__ import annotations

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.split import (
    RESIDUAL_STRATUM,
    Holdout,
    Split,
    SubjectLeakError,
    ensure_patients_per_class,
    strata,
)

FOLDS = 2


def _notes(*subjects: int) -> list[InputNote]:
    return [
        InputNote(f"{subject}-DS-{index}", subject, Language.EN, "text", frozenset())
        for index, subject in enumerate(subjects)
    ]


def _labelled(*patients: tuple[int, set[Diagnosis]]) -> list[InputNote]:
    return [
        InputNote(
            f"{subject}-DS-{index}", subject, Language.EN, "text", frozenset(labels)
        )
        for index, (subject, labels) in enumerate(patients)
    ]


def test_disjoint_folds_pass() -> None:
    """Foldy bez společného pacienta tvoří platný split."""
    split = Split((frozenset({1, 2}), frozenset({3})))

    assert split.folds == (frozenset({1, 2}), frozenset({3}))


def test_shared_subject_is_a_leak() -> None:
    """Pacient ve dvou foldech vyhodí SubjectLeakError se jmenovanými pacienty."""
    folds = (frozenset({1, 2, 5}), frozenset({2, 3}), frozenset({5, 4}))

    with pytest.raises(SubjectLeakError, match=r"Pacienti ve více foldech: 2, 5\.$"):
        Split(folds)


def test_holdout_splits_notes_by_subject_in_input_order() -> None:
    """Test jsou zprávy pacientů foldu, trénink ostatní; pořadí zůstává a fold se přenese."""
    notes = _notes(1, 2, 1, 3, 2)
    split = Split((frozenset({1, 3}), frozenset({2})))

    holdout = split.holdout(notes, 0)

    assert holdout == Holdout(
        fold=0,
        train=(notes[1], notes[4]),
        test=(notes[0], notes[2], notes[3]),
    )
    assert split.holdout(notes, 1).fold == 1
    assert split.holdout(notes, 1).test == (notes[1], notes[4])


@pytest.mark.parametrize("fold", [2, -1])
def test_fold_out_of_range_is_invalid_input(fold: int) -> None:
    """Fold mimo 0..počet-1 je neplatný vstup, záporný index nevybere poslední."""
    split = Split((frozenset({1}), frozenset({2})))

    with pytest.raises(InvalidInputError, match=r"Fold .* 2 foldů"):
        split.holdout(_notes(1, 2), fold)


def test_note_of_subject_outside_split_is_invalid_input() -> None:
    """Zpráva pacienta mimo split nese v hlášce počet, ne identifikátor."""
    split = Split((frozenset({1}), frozenset({2})))

    with pytest.raises(InvalidInputError, match=r"2 zpráv") as raised:
        split.holdout(_notes(1, 2, 987654, 987654), 0)

    assert "987654" not in str(raised.value)


def test_enough_patients_in_each_class_pass() -> None:
    """Každá třída má aspoň tolik pacientů jako foldů."""
    ensure_patients_per_class(
        [True, True, False, False, False], [1, 2, 3, 4, 4], Diagnosis.CKD, FOLDS
    )


def test_class_without_enough_patients_is_invalid_input() -> None:
    """Zpráva jmenuje diagnózu a počty pacientů pozitivních i negativních."""
    with pytest.raises(InvalidInputError, match=r"ckd.*1 pozitivních.*3 negativních"):
        ensure_patients_per_class(
            [True, True, False, False, False], [1, 1, 2, 3, 4], Diagnosis.CKD, FOLDS
        )


def test_class_counts_patients_not_notes() -> None:
    """Pacient s více zprávami se počítá jednou."""
    with pytest.raises(InvalidInputError):
        ensure_patients_per_class(
            [True, True, False, False], [1, 1, 2, 3], Diagnosis.PNEUMONIA, FOLDS
        )


DIABETES_CKD = 0b101
DIABETES = 0b001


def test_patient_without_diagnoses_has_stratum_zero() -> None:
    """Dost pacientů bez diagnóz tvoří třídu 0."""
    notes = _labelled((1, set()), (2, set()), (3, set()))

    assert strata(notes, FOLDS) == (0, 0, 0)


def test_stratum_is_union_of_patient_labels() -> None:
    """Obě zprávy pacienta mají masku sjednocení jeho diagnóz v pořadí Diagnosis."""
    notes = _labelled(
        (1, {Diagnosis.DIABETES}),
        (1, {Diagnosis.CKD}),
        (2, {Diagnosis.DIABETES, Diagnosis.CKD}),
    )

    assert strata(notes, FOLDS) == (DIABETES_CKD, DIABETES_CKD, DIABETES_CKD)


def test_rare_combination_goes_to_residual_stratum() -> None:
    """Kombinace s méně pacienty než foldů tvoří zbytkovou třídu, pokud je jich dost."""
    notes = _labelled(
        (1, {Diagnosis.DIABETES}),
        (2, {Diagnosis.DIABETES}),
        (3, {Diagnosis.CKD}),
        (4, {Diagnosis.COPD}),
    )

    assert strata(notes, FOLDS) == (
        DIABETES,
        DIABETES,
        RESIDUAL_STRATUM,
        RESIDUAL_STRATUM,
    )


def test_small_residual_joins_most_common_combination() -> None:
    """Zbytek menší než foldy převezme nejčastější kombinace, při shodě nižší maska."""
    notes = _labelled(
        (1, {Diagnosis.CKD}),
        (2, {Diagnosis.CKD}),
        (3, {Diagnosis.DIABETES}),
        (4, {Diagnosis.DIABETES}),
        (5, {Diagnosis.COPD}),
    )

    assert strata(notes, FOLDS) == (0b100, 0b100, DIABETES, DIABETES, DIABETES)


def test_strata_do_not_depend_on_note_order() -> None:
    """Permutace zpráv dá stejnou třídu u každé zprávy."""
    notes = _labelled(
        (1, {Diagnosis.CKD}),
        (2, {Diagnosis.CKD}),
        (3, {Diagnosis.DIABETES}),
        (4, {Diagnosis.DIABETES}),
        (5, {Diagnosis.COPD}),
        (5, set()),
        (6, set()),
    )
    reverse = notes[::-1]

    forward = dict(zip((n.note_id for n in notes), strata(notes, FOLDS), strict=True))
    backward = dict(
        zip((n.note_id for n in reverse), strata(reverse, FOLDS), strict=True)
    )

    assert forward == backward
