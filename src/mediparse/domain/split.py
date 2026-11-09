"""Pravidla rozdělení zpráv na trénink a test po pacientech."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.seed import Seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote


RESIDUAL_STRATUM: Final = -1


class SubjectLeakError(ValueError):
    """Jeden pacient je ve více částech splitu."""


class SplitConfig(BaseModel):
    """Parametry splitu sdíleného všemi řádky matice: seed a počet foldů."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: Seed
    folds: Annotated[int, Field(ge=2)]


@dataclass(frozen=True)
class Holdout:
    """Zprávy tréninkové a testovací části jednoho foldu v pořadí vstupu."""

    fold: int
    train: tuple[InputNote, ...]
    test: tuple[InputNote, ...]


@dataclass(frozen=True)
class Split:
    """Disjunktní pacienti foldů evaluace a vývojové části.

    Attributes:
        folds: Množina ``subject_id`` testovací části každého foldu.
        development: Pacienti vyčlenění pro výběr modelů mimo evaluaci.
    """

    folds: tuple[frozenset[SubjectId], ...]
    development: frozenset[SubjectId]

    def __post_init__(self) -> None:
        """Ověří disjunktnost všech částí, aby neplatný split nešel vytvořit.

        Raises:
            SubjectLeakError: Některý pacient je ve více částech splitu.
        """
        seen: set[SubjectId] = set()
        shared: set[SubjectId] = set()
        for fold in (*self.folds, self.development):
            shared |= seen & fold
            seen |= fold
        if shared:
            msg = f"Pacientů ve více částech splitu: {len(shared)}."
            raise SubjectLeakError(msg)

    @classmethod
    def from_parts(cls, parts: Sequence[frozenset[SubjectId]]) -> Split:
        """Vyhradí poslední část pro vývoj, ostatní pro foldy evaluace.

        Args:
            parts: Stratifikované části pacientů v pořadí částí z dělení.

        Returns:
            Split s poslední částí vyčleněnou pro výběr modelů.

        Raises:
            ValueError: Méně než dva foldy a vývojová část jsou chybou kódu.
        """
        if len(parts) < 3:  # ruff: ignore[magic-value-comparison]
            msg = "Split potřebuje aspoň 3 části: dva foldy a vývojovou část."
            raise ValueError(msg)
        # Vývoj má tutéž stratifikaci jako evaluace, bez druhého algoritmu výběru.
        return cls(tuple(parts[:-1]), development=parts[-1])

    def sha256(self) -> str:
        """Otisk přiřazení pacientů do foldů a vývojové části.

        Kontrakt pro každého, kdo otisk počítá z jiného zdroje splitu: SHA-256
        (hex) UTF-8 textu, kde každý pacient je jeden řádek ``subject_id`` TAB
        ``fold`` LF (i poslední řádek končí LF). Čísla jsou desítková bez
        úvodních nul, ``fold`` je pozice foldu v ``folds`` od 0 a řádky jsou
        seřazené vzestupně číselně podle ``subject_id``. Vývojový pacient má
        místo čísla foldu doslovný řetězec ``development``. Prázdná část se
        v otisku neprojeví.

        Returns:
            Hex SHA-256 přiřazení.
        """
        # Číselně, protože nezávislý čtenář souboru (DuckDB, sort -n) řadí celá
        # čísla; textové řazení (10 před 2) by dalo jiný otisk.
        assignment = [
            (subject, str(fold))
            for fold, subjects in enumerate(self.folds)
            for subject in subjects
        ]
        assignment.extend((subject, "development") for subject in self.development)
        lines = "".join(f"{subject}\t{fold}\n" for subject, fold in sorted(assignment))
        return hashlib.sha256(lines.encode()).hexdigest()

    def holdout(self, notes: Sequence[InputNote], fold: int) -> Holdout:
        """Vrátí test pacientů foldu a trénink ostatních foldů, bez vývojových.

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
        self._ensure_covered(notes)
        evaluation = frozenset().union(*self.folds)
        test = tuple(note for note in notes if note.subject_id in self.folds[fold])
        train_subjects = evaluation - self.folds[fold]
        train = tuple(note for note in notes if note.subject_id in train_subjects)
        return Holdout(fold, train, test)

    def development_notes(self, notes: Sequence[InputNote]) -> tuple[InputNote, ...]:
        """Vrátí zprávy vývojových pacientů v pořadí vstupu.

        Args:
            notes: Zprávy, jejichž pacienti musí být všichni ve splitu.

        Returns:
            Zprávy vyčleněné pro výběr modelů mimo evaluaci.

        Raises:
            InvalidInputError: Zpráva patří pacientovi mimo split.
        """
        self._ensure_covered(notes)
        return tuple(note for note in notes if note.subject_id in self.development)

    def _ensure_covered(self, notes: Sequence[InputNote]) -> None:
        covered = frozenset().union(*self.folds, self.development)
        outside = sum(note.subject_id not in covered for note in notes)
        if outside:
            msg = f"Zprávy pacientů mimo split: {outside}."
            raise InvalidInputError(msg)


@dataclass(frozen=True)
class SharedSplit:
    """Split sdílený větvemi a řádky matice spolu s parametry, ze kterých vznikl.

    Attributes:
        config: Seed a počet foldů splitu.
        split: Pacienti každého foldu; foldů je ``config.folds``.
    """

    config: SplitConfig
    split: Split

    def __post_init__(self) -> None:
        """Ověří, že split má tolik foldů, kolik říká config.

        Raises:
            ValueError: Počet foldů splitu se liší od configu (chyba kódu).
        """
        if len(self.split.folds) != self.config.folds:
            msg = f"Config má {self.config.folds} foldů, split {len(self.split.folds)}."
            raise ValueError(msg)


def ensure_enough_patients(subjects: Sequence[SubjectId], parts: int) -> None:
    """Ověří, že pacientů je aspoň tolik jako částí dělení.

    Args:
        subjects: Pacient každé zprávy; pacient s více zprávami se počítá jednou.
        parts: Počet částí dělení.

    Raises:
        InvalidInputError: Pacientů je méně než částí dělení.
    """
    patients = len(set(subjects))
    if patients < parts:
        msg = (
            f"Pacientů je {patients}, dělení do {parts} částí (foldy a vývojová) "
            f"potřebuje aspoň {parts}."
        )
        raise InvalidInputError(msg)


def strata(notes: Sequence[InputNote], parts: int) -> tuple[int, ...]:
    """Určí třídu stratifikace každé zprávy z kombinace diagnóz jejího pacienta.

    Kombinace diagnóz pacienta (sjednocení labelů všech jeho zpráv) je bitová
    maska v pořadí ``Diagnosis``; na pořadí čísel nezáleží, jen na identitě třídy.
    Předpokládá aspoň tolik různých pacientů jako částí dělení; jinak by zbytková
    třída vyvolala varování sklearn.

    Args:
        notes: Zprávy korpusu.
        parts: Počet částí dělení.

    Returns:
        Třída každé zprávy v pořadí vstupu.
    """
    stratum = _stratum_of(_combinations(notes), parts)
    return tuple(stratum[note.subject_id] for note in notes)


def _combinations(notes: Sequence[InputNote]) -> dict[SubjectId, int]:
    order = {diagnosis: 1 << index for index, diagnosis in enumerate(Diagnosis)}
    combinations: dict[SubjectId, int] = {}
    for note in notes:
        mask = sum(order[diagnosis] for diagnosis in note.labels)
        combinations[note.subject_id] = combinations.get(note.subject_id, 0) | mask
    return combinations


def _stratum_of(combinations: dict[SubjectId, int], parts: int) -> dict[SubjectId, int]:
    """Přiřadí pacientům třídy stratifikace se sloučením vzácných kombinací.

    Args:
        combinations: Bitová maska kombinace diagnóz každého pacienta.
        parts: Počet částí dělení.

    Returns:
        Třída stratifikace každého pacienta.
    """
    # sklearn StratifiedGroupKFold ohlásí varováním třídu s méně členy než částí,
    # a ta se nedá rozložit do všech částí; vzácné kombinace proto sloučíme.
    patients = Counter(combinations.values())
    rare = {mask for mask, count in patients.items() if count < parts}
    frequent = {mask: count for mask, count in patients.items() if mask not in rare}
    residual_patients = sum(patients[mask] for mask in rare)
    if frequent and 0 < residual_patients < parts:
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
    positive = {
        subject for label, subject in zip(labels, subjects, strict=True) if label
    }
    negative = {
        subject for label, subject in zip(labels, subjects, strict=True) if not label
    }
    if min(len(positive), len(negative)) < folds:
        msg = (
            f"Diagnóza {diagnosis}: {len(positive)} pozitivních a {len(negative)} "
            f"negativních pacientů, odložení potřebuje v každé třídě aspoň {folds}."
        )
        raise InvalidInputError(msg)
