"""Vyhodnocení pokusu přes validační foldy, zápis výsledků a oznámení dokončení."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.stage_tuning import StageFoldResult
from mediparse.domain.tuning import fold_max_f1, macro_max_f1

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.application.ports.experiment import StageProgress, TrialScorer
    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import StageTrialRecord
    from mediparse.domain.tuning import MaxF1, ParamValue, ValidationScores


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialRun:
    """Skóre, prahy a stav všech vyhodnocených foldů pokusu."""

    scores: tuple[ValidationScores, ...]
    f1: tuple[dict[Diagnosis, MaxF1], ...]
    value: float
    converged: bool


def score_trial(
    scorer: TrialScorer,
    params: Mapping[str, ParamValue],
    folds: Sequence[Holdout],
) -> TrialRun:
    """Spočte průměr macro max-F1 všech foldů.

    Returns:
        Skóre, prahy, společný cíl a konvergence všech fitů.
    """
    scores = scorer.score_folds(params, folds)
    f1 = tuple(
        fold_max_f1(fold.test, scored)
        for fold, scored in zip(folds, scores, strict=True)
    )
    return TrialRun(
        scores,
        f1,
        sum(map(macro_max_f1, f1)) / len(folds),
        all(s.converged for s in scores),
    )


def trial_results(
    run: TrialRun, folds: Sequence[Holdout]
) -> tuple[StageFoldResult, ...]:
    """Sestaví buňky diagnóz a foldů pro trvalý checkpoint.

    Returns:
        Predikce, prahy a velikosti reprezentace ve stabilním pořadí.
    """
    return tuple(
        StageFoldResult(
            fold=fold.fold,
            diagnosis=diagnosis,
            note_ids=tuple(note.note_id for note in fold.test),
            labels=tuple(note.has_label(diagnosis) for note in fold.test),
            scores=run.scores[i].scores[diagnosis],
            max_f1=run.f1[i][diagnosis],
            converged=run.scores[i].converged,
            n_features=run.scores[i].n_features,
        )
        for i, fold in enumerate(folds)
        for diagnosis in Diagnosis
    )


def write_trial(
    progress: StageProgress,
    record: StageTrialRecord,
    results: Sequence[StageFoldResult],
) -> None:
    """Zapíše hotový pokus do průběhu a oznámí jeho dokončení."""
    progress.write(record, results)
    logger.info(
        "Pokus %d COMPLETE: macro-F1 %.6f, čas %.3f s.",
        record.number,
        record.value,
        record.seconds,
    )
