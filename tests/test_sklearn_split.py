"""Adaptér splitu scikit-learn: pacienti po foldech stratifikovaní podle třídy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.domain.split import strata
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from tests.model_support import separable_notes, split_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.split import Split

FOLDS: Final = 3
SEEDS: Final = range(10)
IMBALANCED_PATIENTS: Final = 30
POSITIVE_PATIENTS: Final = 6
EXPECTED_POSITIVES: Final = POSITIVE_PATIENTS // FOLDS
assert POSITIVE_PATIENTS % FOLDS == 0


def _imbalanced_notes() -> list[InputNote]:
    """Třicet pacientů, každý pátý s diabetem, 1 až 4 zprávy nepravidelně.

    Returns:
        Zprávy po pacientech v pořadí pacientů.
    """
    notes = []
    for patient in range(1, IMBALANCED_PATIENTS + 1):
        labels = frozenset({Diagnosis.DIABETES} if patient % 5 == 0 else ())
        notes.extend(
            InputNote(f"{patient}-DS-{visit}", patient, Language.EN, "text", labels)
            for visit in range(1 + (patient * 7) % 4)
        )
    return notes


def _split(notes: Sequence[InputNote], *, folds: int, seed: int) -> Split:
    return StratifiedGroupFolds().split(
        [note.subject_id for note in notes],
        strata(notes, folds),
        split_config(folds=folds, seed=seed),
    )


@pytest.mark.parametrize(
    "notes",
    [separable_notes(12), _imbalanced_notes()],
    ids=["alternating", "imbalanced"],
)
def test_every_patient_is_in_exactly_one_fold(notes: list[InputNote]) -> None:
    """Foldy se nepřekrývají, dají všechny pacienty a je jich tolik, kolik se chtělo."""
    split = _split(notes, folds=FOLDS, seed=0)

    patients = [subject for fold in split.folds for subject in fold]
    assert len(split.folds) == FOLDS
    assert set(patients) == {note.subject_id for note in notes}


def test_split_is_deterministic_for_seed() -> None:
    """Stejný seed dá stejný split a některý jiný seed jiný."""
    notes = separable_notes(12)

    first = _split(notes, folds=2, seed=3)

    assert _split(notes, folds=2, seed=3) == first
    assert any(_split(notes, folds=2, seed=seed) != first for seed in range(4, 12))


@pytest.mark.parametrize("seed", SEEDS)
def test_positive_patients_are_spread_evenly_over_folds(seed: int) -> None:
    """Šest pozitivních pacientů je ve třech foldech po dvou, což náhodné dělení nesplní."""
    notes = _imbalanced_notes()
    positive = {note.subject_id for note in notes if note.labels}

    split = _split(notes, folds=FOLDS, seed=seed)

    assert [len(fold & positive) for fold in split.folds] == [
        EXPECTED_POSITIVES
    ] * FOLDS


def test_rare_combination_does_not_warn() -> None:
    """Vzácná kombinace diagnóz se sloučí a sklearn nevaruje (varování jsou v testech chyba)."""
    notes = [
        InputNote(f"{patient}-DS-0", patient, Language.EN, "text", frozenset(labels))
        for patient, labels in enumerate(
            [{Diagnosis.DIABETES}] * 6
            + [{Diagnosis.CKD}] * 6
            + [set()] * 6
            + [{Diagnosis.COPD, Diagnosis.CKD}],
            start=1,
        )
    ]

    split = _split(notes, folds=FOLDS, seed=0)

    assert sum(len(fold) for fold in split.folds) == len(notes)
