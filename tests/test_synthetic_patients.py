"""Syntetičtí pacienti: počty zpráv, identifikátory, dědění chronických labelů a tolerance prevalencí."""

from __future__ import annotations

from collections import Counter
from math import isclose
from random import Random
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.labels import Diagnosis, LabelModel
from mediparse.domain.note import subject_of
from mediparse.domain.synthetic_patients import (
    PatientModel,
    PrevalenceNotReachedError,
    continuation_ratio,
    sample_notes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.synthetic_patients import SyntheticNote

PREVALENCE = {
    Diagnosis.DIABETES: 0.27,
    Diagnosis.CKD: 0.18,
    Diagnosis.HEART_FAILURE: 0.17,
    Diagnosis.ATRIAL_FIBRILLATION: 0.17,
    Diagnosis.AKI: 0.16,
}
CORPUS = 200
LARGE = 40000
TOLERANCE = 0.02
FIRST_SUBJECT = 90000001
SINGLE_NOTE = 0.587
MAX_NOTES = 8
MEAN_MULTIPLE = 4.07
SHARE_SLACK = 0.015
MEAN_SLACK = 0.1
SEED = 20260929


def _labels(tolerance: float = TOLERANCE) -> LabelModel:
    return LabelModel(
        prevalence=PREVALENCE,
        conditional={
            first: {
                second: 1.0 if first is second else PREVALENCE[second]
                for second in Diagnosis
            }
            for first in Diagnosis
        },
        prevalence_tolerance=tolerance,
        ipf_tolerance=1e-12,
        ipf_max_passes=500,
    )


def _patients(notes: int = CORPUS, attempts: int = 1000) -> PatientModel:
    return PatientModel(
        notes=notes,
        first_subject_id=FIRST_SUBJECT,
        single_note_probability=SINGLE_NOTE,
        max_notes=MAX_NOTES,
        mean_notes_when_multiple=MEAN_MULTIPLE,
        max_attempts=attempts,
    )


def _sample(
    notes: int = CORPUS, tolerance: float = TOLERANCE
) -> tuple[SyntheticNote, ...]:
    return sample_notes(_patients(notes), _labels(tolerance), Random(SEED))


def _by_subject(notes: Sequence[SyntheticNote]) -> dict[int, list[SyntheticNote]]:
    groups: dict[int, list[SyntheticNote]] = {}
    for note in notes:
        groups.setdefault(note.subject_id, []).append(note)
    return groups


def test_same_seed_gives_same_notes() -> None:
    """Vzorkovač je deterministický: stejný seed dává stejné zprávy, jiný jiné."""
    other = sample_notes(_patients(), _labels(), Random(SEED + 1))

    assert _sample() == _sample()
    assert _sample() != other


def test_corpus_size_and_prevalences_hold() -> None:
    """Korpus má přesně zadaný počet zpráv a každá prevalence leží v toleranci."""
    notes = _sample()

    assert len(notes) == CORPUS
    for diagnosis, prevalence in PREVALENCE.items():
        share = sum(diagnosis in note.labels for note in notes) / len(notes)
        assert abs(share - prevalence) <= TOLERANCE


def test_note_ids_follow_mimic_scheme_per_patient() -> None:
    """Note_id nese subject_id pacienta a pořadí zprávy od jedné."""
    notes = _sample()

    for subject, patient_notes in _by_subject(notes).items():
        assert [note.note_id for note in patient_notes] == [
            f"{subject}-DS-{order}" for order in range(1, len(patient_notes) + 1)
        ]
        assert all(subject_of(note.note_id) == str(subject) for note in patient_notes)
    assert min(note.subject_id for note in notes) == FIRST_SUBJECT


def test_patient_has_at_most_max_notes() -> None:
    """Žádný pacient nemá víc zpráv, než dovoluje horní mez."""
    counts = Counter(note.subject_id for note in _sample(LARGE, 1.0))

    assert max(counts.values()) <= MAX_NOTES


def test_chronic_labels_are_shared_by_patient_notes() -> None:
    """Chronické diagnózy platí pro všechny zprávy pacienta, AKI se může lišit."""
    for patient_notes in _by_subject(_sample(LARGE, 1.0)).values():
        chronic = {frozenset(d for d in n.labels if d.chronic) for n in patient_notes}
        assert len(chronic) == 1


def test_note_counts_follow_the_model() -> None:
    """Na velkém vzorku sedí podíl pacientů s jedinou zprávou i průměr ostatních."""
    counts = list(Counter(note.subject_id for note in _sample(LARGE, 1.0)).values())
    complete = counts[:-1]
    multiple = [count for count in complete if count > 1]

    assert abs(complete.count(1) / len(complete) - SINGLE_NOTE) <= SHARE_SLACK
    assert abs(sum(multiple) / len(multiple) - MEAN_MULTIPLE) <= MEAN_SLACK


def test_ratio_gives_requested_truncated_mean() -> None:
    """Kvocient geometrického rozdělení dává po oříznutí přesně požadovaný průměr."""
    ratio = continuation_ratio(_patients())
    counts = range(2, MAX_NOTES + 1)
    weights = [ratio ** (k - 2) for k in counts]
    mean = sum(k * w for k, w in zip(counts, weights, strict=True)) / sum(weights)

    assert 0.0 < ratio < 1.0
    assert isclose(mean, MEAN_MULTIPLE, abs_tol=1e-9)


def test_unreachable_tolerance_fails_loudly() -> None:
    """Nedosažitelná tolerance skončí výjimkou, ne nekonečnou smyčkou."""
    with pytest.raises(PrevalenceNotReachedError):
        sample_notes(_patients(attempts=3), _labels(0.0), Random(SEED))
