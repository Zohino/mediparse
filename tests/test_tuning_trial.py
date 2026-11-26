"""Ruční orákulum cíle pokusu přes foldy."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from mediparse.application.tuning_trial import score_trial
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.split import Holdout
from mediparse.domain.tuning import ParamValue, ValidationScores
from tests.screening_support import note

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def test_score_trial_averages_fold_macro_f1() -> None:
    """Konstantní skóre s jedním pozitivním ze tří dává 0,5, oddělení 1: průměr 0,75."""
    notes = tuple(
        replace(
            note(i, 0, positive=i == 1),
            labels=frozenset(Diagnosis) if i == 1 else frozenset(),
        )
        for i in range(1, 4)
    )
    folds = tuple(Holdout(fold=i, train=notes, test=notes) for i in range(2))

    class Scorer:
        @staticmethod
        def score_folds(
            params: Mapping[str, ParamValue], folds: Sequence[Holdout]
        ) -> tuple[ValidationScores, ...]:
            del params, folds
            return tuple(
                ValidationScores(
                    dict.fromkeys(Diagnosis, scores), converged=True, n_features=1
                )
                for scores in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
            )

    assert score_trial(Scorer(), {}, folds).value == pytest.approx(0.75)
