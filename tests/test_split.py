"""Pravidla splitu: disjunktní pacienti a dost pacientů v každé třídě."""

from __future__ import annotations

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import (
    Holdout,
    SubjectLeakError,
    ensure_disjoint_subjects,
    ensure_patients_per_class,
)

FOLDS = 2


def _notes(*subjects: int) -> list[InputNote]:
    return [
        InputNote(f"{subject}-DS-{index}", subject, "en", "text", frozenset())
        for index, subject in enumerate(subjects)
    ]


def test_disjoint_subjects_pass() -> None:
    """Pacienti v tréninku a testu se neprotínají."""
    notes = _notes(1, 1, 2, 3)

    ensure_disjoint_subjects(notes, Holdout(train=(0, 1), test=(2, 3)))


def test_shared_subject_is_a_leak() -> None:
    """Pacient v obou částech vyhodí SubjectLeakError se jmenovaným pacientem."""
    notes = _notes(1, 2, 1)

    with pytest.raises(SubjectLeakError, match="1"):
        ensure_disjoint_subjects(notes, Holdout(train=(0, 1), test=(2,)))


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
