"""Křížová validace po pacientech uvnitř dodané množiny zpráv."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import ensure_min_cell
from mediparse.domain.split import Split, ensure_patients_per_class, strata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        PatientNoteTable,
        PatientPartitioner,
        SplitSource,
    )
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout, SharedSplit


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


def development_notes(
    table: PatientNoteTable, split: SplitSource
) -> tuple[SharedSplit, tuple[InputNote, ...]]:
    """Načte vývojové zprávy po ověření úplnosti splitu.

    Returns:
        Sdílený split a zprávy jeho vývojových pacientů.
    """
    shared = split.read()
    shared.split.ensure_covers(table.subjects())
    return shared, table.read_patients(shared.split.development)


def development_sample(
    notes: Sequence[InputNote], parts: int, seed: Seed, partitioner: PatientPartitioner
) -> Sequence[InputNote]:
    """Vybere společný pacientský vzorek předkola.

    Returns:
        Celá množina pro jednu část, jinak první část stratifikace.
    """
    return (
        notes
        if parts == 1
        else patient_holdouts(notes, parts, seed, partitioner)[0].test
    )
