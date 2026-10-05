"""Doména tréninku smoketestu: neplatný config a invariant disjunktních pacientů."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import InputNote
from mediparse.domain.smoketest_training import (
    Holdout,
    SubjectLeakError,
    TrainingConfig,
    ensure_disjoint_subjects,
    ensure_patients_per_class,
)

FOLDS = 2
VALID = {"diagnosis": "diabetes", "seed": 0, "folds": FOLDS, "regularization": 1.0}


def _notes(*subjects: int) -> list[InputNote]:
    return [
        InputNote(f"{subject}-DS-{index}", subject, "en", "text", frozenset())
        for index, subject in enumerate(subjects)
    ]


def test_valid_config_is_parsed() -> None:
    """Platný JSON dá config s výčtem diagnózy."""
    config = TrainingConfig.model_validate_json(
        '{"diagnosis": "diabetes", "seed": 0, "folds": 2, "regularization": 1.0}'
    )

    assert config.diagnosis is Diagnosis.DIABETES
    assert config.folds == FOLDS


@pytest.mark.parametrize(
    "override",
    [
        {"folds": 1},
        {"regularization": 0.0},
        {"regularization": -1.0},
        {"diagnosis": "asthma"},
        {"unexpected": 1},
    ],
)
def test_invalid_config_is_rejected(override: dict[str, object]) -> None:
    """Málo foldů, nekladné C, neznámá diagnóza i extra klíč config odmítnou."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({**VALID, **override})


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
            [True, True, False, False], [1, 1, 2, 3], Diagnosis.AKI, FOLDS
        )
