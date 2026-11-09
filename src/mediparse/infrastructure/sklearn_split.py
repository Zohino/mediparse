"""Rozdělení pacientů do částí ve scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn.model_selection import StratifiedGroupKFold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.identity import SubjectId
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
