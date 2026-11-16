"""Křížová validace po pacientech uvnitř dodané množiny zpráv."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import ensure_min_cell
from mediparse.domain.split import Split, ensure_patients_per_class, strata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import PatientPartitioner
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout


def patient_holdouts(
    notes: Sequence[InputNote], folds: int, seed: Seed, partitioner: PatientPartitioner
) -> tuple[Holdout, ...]:
    """Rozdělí pacienty se společnou stratifikací pěti diagnóz a ověří validace.

    Returns:
        Foldy, v nichž je každý pacient právě jednou ve validaci.

    Raises:
        InvalidInputError: Třída nemá dost pacientů nebo validace nesplní MIN_CELL.
    """
    subjects = [note.subject_id for note in notes]
    for diagnosis in Diagnosis:
        ensure_patients_per_class(
            [note.has_label(diagnosis) for note in notes], subjects, diagnosis, folds
        )
    parts = partitioner.partition(subjects, strata(notes, folds), folds, seed)
    split = Split(parts, development=frozenset())
    holdouts = tuple(split.holdout(notes, fold) for fold in range(folds))
    for holdout in holdouts:
        for diagnosis in Diagnosis:
            ensure_min_cell(
                diagnosis, [note.has_label(diagnosis) for note in holdout.test]
            )
    return holdouts
