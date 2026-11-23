"""Rozdělení pacientů do částí ve scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from mediparse.domain.split import ensure_enough_patients, strata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed


@dataclass(frozen=True)
class StratifiedGroupFolds:
    """Stratifikované rozdělení pacientů do částí; pacient je celý v jedné."""

    @staticmethod
    def partition(
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        parts: int,
        seed: Seed,
    ) -> tuple[frozenset[SubjectId], ...]:
        """Rozloží pacienty do částí podle třídy stratifikace jejich zpráv.

        Args:
            subjects: Pacient každé zprávy.
            strata: Třída stratifikace každé zprávy ze stejného pořadí.
            parts: Počet částí dělení.
            seed: Seed náhodného rozložení.

        Returns:
            Množina pacientů testovací části každého dělení.
        """
        splitter = StratifiedGroupKFold(n_splits=parts, shuffle=True, random_state=seed)
        partitions = splitter.split(list(subjects), list(strata), list(subjects))
        return tuple(
            frozenset(subjects[index] for index in test) for _, test in partitions
        )


def threshold_splits(
    labels: Sequence[bool], subjects: Sequence[SubjectId], folds: int, seed: Seed
) -> list[tuple[ndarray, ndarray]]:
    """Převede stratifikované části pacientů na indexy zpráv pro vnitřní CV.

    Returns:
        Páry indexů tréninku a testu; každý pacient náleží jedinému testu.
    """
    # Předpočítané indexy nevyžadují globální enable_metadata_routing pro groups.
    partitions = StratifiedGroupFolds.partition(
        subjects, [int(label) for label in labels], folds, seed
    )
    return [
        (
            np.array(
                [i for i, subject in enumerate(subjects) if subject not in part],
                dtype=int,
            ),
            np.array(
                [i for i, subject in enumerate(subjects) if subject in part], dtype=int
            ),
        )
        for part in partitions
    ]


def holdout_mask(notes: Sequence[InputNote], parts: int, seed: Seed) -> ndarray:
    """Vybere stratifikovanou odloženou část celých pacientů.

    Returns:
        Boolovská maska zpráv první části rozdělení.

    Raises:
        InvalidInputError: Pacientů je méně než částí dělení.
    """
    subjects = [note.subject_id for note in notes]
    ensure_enough_patients(subjects, parts)
    held = StratifiedGroupFolds.partition(subjects, strata(notes, parts), parts, seed)[
        0
    ]
    return np.array([subject in held for subject in subjects], dtype=bool)
