"""Pravidla rozdělení zpráv na trénink a test po pacientech."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote


RESIDUAL_STRATUM: Final = -1


class SubjectLeakError(ValueError):
    """Jeden pacient je ve dvou foldech."""


@dataclass(frozen=True)
class Holdout:
    """Zprávy tréninkové a testovací části jednoho foldu v pořadí vstupu."""

    fold: int
    train: tuple[InputNote, ...]
    test: tuple[InputNote, ...]


@dataclass(frozen=True)
class Split:
    """Pacienti každého foldu; žádný pacient není ve dvou foldech.

    Attributes:
        folds: Množina ``subject_id`` testovací části každého foldu.
    """

    folds: tuple[frozenset[SubjectId], ...]

    def __post_init__(self) -> None:
        """Ověří disjunktnost foldů, aby neplatný split nešel vytvořit.

        Raises:
            SubjectLeakError: Některý pacient je ve dvou foldech.
        """
        seen: set[SubjectId] = set()
        shared: set[SubjectId] = set()
        for fold in self.folds:
            shared |= seen & fold
            seen |= fold
        if shared:
            msg = f"Pacientů ve více foldech: {len(shared)}."
            raise SubjectLeakError(msg)

    def holdout(self, notes: Sequence[InputNote], fold: int) -> Holdout:
        """Rozdělí zprávy na test pacientů foldu a trénink ostatních.

        Args:
            notes: Zprávy, jejichž pacienti musí být všichni ve splitu.
            fold: Číslo testovacího foldu.

        Returns:
            Zprávy tréninku a testu v pořadí vstupu.

        Raises:
            InvalidInputError: Fold je mimo rozsah nebo zpráva patří pacientovi
                mimo split.
        """
        if not 0 <= fold < len(self.folds):
            msg = f"Fold {fold} neexistuje, počet foldů splitu: {len(self.folds)}."
            raise InvalidInputError(msg)
        covered = frozenset().union(*self.folds)
        outside = sum(note.subject_id not in covered for note in notes)
        if outside:
            msg = f"Zprávy pacientů mimo split: {outside}."
            raise InvalidInputError(msg)
        test = tuple(note for note in notes if note.subject_id in self.folds[fold])
        train = tuple(note for note in notes if note.subject_id not in self.folds[fold])
        return Holdout(fold, train, test)


def strata(notes: Sequence[InputNote], folds: int) -> tuple[int, ...]:
    """Určí třídu stratifikace každé zprávy z kombinace diagnóz jejího pacienta.

    Kombinace diagnóz pacienta (sjednocení labelů všech jeho zpráv) je bitová
    maska v pořadí ``Diagnosis``; na pořadí čísel nezáleží, jen na identitě třídy. Předpokládá aspoň tolik
    pacientů jako foldů (use case to zajistí ``ensure_patients_per_class``); jinak
    by zbytková třída vyvolala varování sklearn.

    Args:
        notes: Zprávy korpusu.
        folds: Počet foldů splitu.

    Returns:
        Třída každé zprávy v pořadí vstupu.
    """
    stratum = _stratum_of(_combinations(notes), folds)
    return tuple(stratum[note.subject_id] for note in notes)


def _combinations(notes: Sequence[InputNote]) -> dict[SubjectId, int]:
    order = {diagnosis: 1 << index for index, diagnosis in enumerate(Diagnosis)}
    combinations: dict[SubjectId, int] = {}
    for note in notes:
        mask = sum(order[diagnosis] for diagnosis in note.labels)
        combinations[note.subject_id] = combinations.get(note.subject_id, 0) | mask
    return combinations


def _stratum_of(combinations: dict[SubjectId, int], folds: int) -> dict[SubjectId, int]:
    # sklearn StratifiedGroupKFold ohlásí varováním třídu s méně členy než foldů,
    # a ta se stejně nedá rozložit do všech foldů; vzácné kombinace proto sloučíme.
    patients = Counter(combinations.values())
    rare = {mask for mask, count in patients.items() if count < folds}
    frequent = {mask: count for mask, count in patients.items() if mask not in rare}
    residual_patients = sum(patients[mask] for mask in rare)
    if frequent and 0 < residual_patients < folds:
        # Sám by zbytek varování vyvolal, proto ho převezme nejčastější kombinace.
        target = min(frequent, key=lambda mask: (-frequent[mask], mask))
        return {
            subject: target if mask in rare else mask
            for subject, mask in combinations.items()
        }
    return {
        subject: RESIDUAL_STRATUM if mask in rare else mask
        for subject, mask in combinations.items()
    }


def ensure_patients_per_class(
    labels: Sequence[bool],
    subjects: Sequence[SubjectId],
    diagnosis: Diagnosis,
    folds: int,
) -> None:
    """Ověří, že každá třída diagnózy má aspoň tolik pacientů jako foldů.

    Args:
        labels: Label diagnózy každé zprávy.
        subjects: Pacient každé zprávy.
        diagnosis: Diagnóza, na které se trénuje.
        folds: Počet foldů odložení.

    Raises:
        InvalidInputError: Některá třída má méně pacientů než foldů.
    """
    positive = {group for label, group in zip(labels, subjects, strict=True) if label}
    negative = {
        group for label, group in zip(labels, subjects, strict=True) if not label
    }
    if min(len(positive), len(negative)) < folds:
        msg = (
            f"Diagnóza {diagnosis}: {len(positive)} pozitivních a {len(negative)} "
            f"negativních pacientů, odložení potřebuje v každé třídě aspoň {folds}."
        )
        raise InvalidInputError(msg)
