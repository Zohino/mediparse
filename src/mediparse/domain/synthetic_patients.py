"""Losování pacientů syntetického korpusu: kolik zpráv pacient má a jaké labely nesou."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.diagnosis import ACUTE_DIAGNOSIS
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import (
    LabelModel,
    LabelSet,
    acute_given_chronic,
    label_shares,
    max_entropy_joint,
    prevalence_outliers,
)
from mediparse.domain.note import note_id
from mediparse.domain.synthetic_note import SyntheticNote
from mediparse.domain.synthetic_plan import MIN_MULTIPLE_NOTES, PatientModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from random import Random

    from mediparse.domain.identity import SubjectId


_BISECTION_STEPS: Final = 100


@dataclass(frozen=True)
class _PatientDistribution:
    """Rozdělení, z nichž losuje každý tah: společné labely, akutní diagnóza podle chronických a kvocient počtu zpráv."""

    joint: Mapping[LabelSet, float]
    acute: Mapping[LabelSet, float]
    ratio: float


class PrevalenceNotReachedError(InvalidInputError):
    """Žádný tah v povoleném počtu pokusů nedal prevalence v toleranci."""


def continuation_ratio(model: PatientModel) -> float:
    """Kvocient posunutě geometrického rozdělení počtu zpráv, oříznutého na ``max_notes``.

    Returns:
        Kvocient, při němž má oříznuté rozdělení požadovaný průměr.
    """
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        ratio = (low + high) / 2
        if _mean_count(ratio, model.max_notes) < model.mean_notes_when_multiple:
            low = ratio
        else:
            high = ratio
    return (low + high) / 2


def sample_notes(
    patients: PatientModel, labels: LabelModel, rng: Random
) -> tuple[SyntheticNote, ...]:
    """Vylosuje zprávy korpusu; tah opakuje, dokud jsou prevalence v toleranci.

    Returns:
        Zprávy v pořadí pacientů, každá s note_id ve skladbě MIMIC-IV-Note.

    Raises:
        PrevalenceNotReachedError: Tolerance nebyla dosažena v povoleném počtu pokusů.
    """
    joint = max_entropy_joint(labels)
    distribution = _PatientDistribution(
        joint, acute_given_chronic(joint), continuation_ratio(patients)
    )
    for _ in range(patients.max_attempts):
        notes = _draw(patients, distribution, rng)
        if _within_tolerance(notes, labels):
            return notes
    msg = f"Prevalence nevyšly do tolerance ani za {patients.max_attempts} pokusů."
    raise PrevalenceNotReachedError(msg)


def _draw(
    patients: PatientModel, distribution: _PatientDistribution, rng: Random
) -> tuple[SyntheticNote, ...]:
    notes: list[SyntheticNote] = []
    subject = patients.first_subject_id
    while len(notes) < patients.notes:
        count = min(
            _note_count(patients, distribution.ratio, rng),
            patients.notes - len(notes),
        )
        notes.extend(_patient_notes(subject, count, distribution, rng))
        subject += 1
    return tuple(notes)


def _note_count(patients: PatientModel, ratio: float, rng: Random) -> int:
    if rng.random() < patients.single_note_probability:
        return 1
    counts, weights = _count_weights(ratio, patients.max_notes)
    return rng.choices(counts, weights=weights)[0]


def _patient_notes(
    subject: SubjectId,
    count: int,
    distribution: _PatientDistribution,
    rng: Random,
) -> list[SyntheticNote]:
    cells = list(distribution.joint)
    first = rng.choices(cells, weights=[distribution.joint[cell] for cell in cells])[0]
    chronic = frozenset(d for d in first if d.chronic)
    later = [
        chronic | {ACUTE_DIAGNOSIS}
        if rng.random() < distribution.acute[chronic]
        else chronic
        for _ in range(count - 1)
    ]
    return [
        SyntheticNote(note_id(subject, order), subject, labels)
        for order, labels in enumerate([first, *later], start=1)
    ]


def _within_tolerance(notes: Sequence[SyntheticNote], labels: LabelModel) -> bool:
    shares = label_shares([note.labels for note in notes])
    return not prevalence_outliers(shares, labels)


def _mean_count(ratio: float, max_notes: int) -> float:
    counts, weights = _count_weights(ratio, max_notes)
    return sum(k * w for k, w in zip(counts, weights, strict=True)) / sum(weights)


def _count_weights(ratio: float, max_notes: int) -> tuple[range, list[float]]:
    counts = range(MIN_MULTIPLE_NOTES, max_notes + 1)
    return counts, [ratio ** (k - MIN_MULTIPLE_NOTES) for k in counts]
