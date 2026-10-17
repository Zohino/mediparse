"""Rozdělení pacientů do foldů ve scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn.model_selection import StratifiedGroupKFold

from mediparse.domain.split import Split, SplitConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.identity import SubjectId


@dataclass(frozen=True)
class StratifiedGroupFolds:
    """Stratifikované rozdělení pacientů do foldů; pacient je celý v jednom."""

    @staticmethod
    def split(
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        config: SplitConfig,
    ) -> Split:
        """Rozloží pacienty do foldů podle třídy stratifikace jejich zpráv.

        Args:
            subjects: Pacient každé zprávy.
            strata: Třída stratifikace každé zprávy ze stejného pořadí.
            config: Seed a počet foldů splitu.

        Returns:
            Split, jehož fold k je množina pacientů testovací části k-tého dělení.
        """
        splitter = StratifiedGroupKFold(
            n_splits=config.folds, shuffle=True, random_state=config.seed
        )
        parts = splitter.split(list(subjects), list(strata), list(subjects))
        return Split(
            tuple(frozenset(subjects[index] for index in test) for _, test in parts)
        )
