"""Sdílený protokol vyhodnocení pokusu přes validační foldy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.tuning import fold_max_f1

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import ParameterSearch, TrialScorer
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.split import Holdout
    from mediparse.domain.tuning import MaxF1, Trial, ValidationScores


@dataclass(frozen=True)
class TrialRun:
    """Skóre, prahy a stav všech vyhodnocených foldů pokusu."""

    scores: tuple[ValidationScores, ...]
    f1: tuple[dict[Diagnosis, MaxF1], ...]
    value: float
    converged: bool
    pruned: bool


def run_trial(
    search: ParameterSearch,
    scorer: TrialScorer,
    trial: Trial,
    folds: Sequence[Holdout],
) -> TrialRun:
    """Vyhodnotí foldy a uzavře pokus společným cílem.

    Returns:
        Výsledek včetně foldů dokončených před prořezáním.
    """
    scores = []
    f1 = []
    values = []
    converged = True
    pruned = False
    for step, fold in enumerate(folds):
        scored = scorer.score(trial.params, fold)
        scores.append(scored)
        metrics = fold_max_f1(fold.test, scored)
        f1.append(metrics)
        values.append(sum(item.value for item in metrics.values()) / len(metrics))
        value = sum(values) / len(values)
        converged &= scored.converged
        if step + 1 < len(folds) and search.should_prune(trial, value, step):
            search.prune(trial)
            pruned = True
            break
    if not pruned:
        search.complete(trial, value)
    return TrialRun(tuple(scores), tuple(f1), value, converged, pruned)
