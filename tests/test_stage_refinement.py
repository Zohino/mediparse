"""Výběr rodin předkola a doladění na vývojové části."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import log10
from typing import TYPE_CHECKING

import pytest

from mediparse.application.stage_refinement import StageRefinement
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import SharedSplit, Split, SplitConfig
from mediparse.domain.stage_refinement import RefinementConfig
from mediparse.domain.stage_tuning import (
    StageTrialRecord,
    best_trial,
    family_candidates,
)
from mediparse.domain.tuning import MaxF1, ValidationScores
from mediparse.infrastructure.optuna_search import OptunaStageSearch
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.stage_trials_table import ParquetStageCheckpoint
from tests.screening_support import note
from tests.stage_support import FixedSearch, MemoryProgress, StepClock, stage_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.input_note import InputNote
    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue


def _record(number: int, branch: str, value: float) -> StageTrialRecord:
    return StageTrialRecord(
        number,
        {"branch": branch},
        default=False,
        value=value,
        converged=True,
        seconds=1.0,
    )


def test_candidates_match_r_rule() -> None:
    """Příklad R A2: a b a b c c, při shodě vítězí nižší číslo."""
    families = {"a": "logreg", "b": "linear_svm", "c": "nb_svm"}
    records = tuple(
        _record(i, f"word/{families[family]}", value)
        for i, (family, value) in enumerate(
            zip("ababcc", (0.80, 0.78, 0.82, 0.82, 0.70, 0.70), strict=True)
        )
    )
    assert tuple(r.number for r in family_candidates(records)) == (2, 3)
    assert tuple(
        r.number
        for r in family_candidates(tuple(replace(r, value=0.8) for r in records))
    ) == (0, 1)


def test_candidates_group_by_family_not_branch() -> None:
    """Dvě reprezentace NB-SVM obsadí jen jedno místo kontroly."""
    records = (
        _record(0, "word/nb_svm", 0.9),
        _record(1, "char_wb/nb_svm", 0.8),
        _record(2, "word/linear_svm", 0.7),
    )
    assert tuple(r.number for r in family_candidates(records)) == (0, 2)
    with pytest.raises(InvalidInputError, match="dvě rodiny"):
        family_candidates(records[:2])


@dataclass(frozen=True)
class _Table:
    @staticmethod
    def subjects() -> frozenset[int]:
        return frozenset(range(90000001, 90000101))

    def read(self) -> tuple[InputNote, ...]:
        return self.read_patients(self.subjects())

    @staticmethod
    def read_patients(subjects: frozenset[int]) -> tuple[InputNote, ...]:
        return tuple(
            replace(
                note(i, 0, positive=True),
                labels=frozenset(Diagnosis) if i % 5 == 0 else frozenset(),
            )
            for i in sorted(subjects)
        )


@dataclass(frozen=True)
class _Split:
    @staticmethod
    def read() -> SharedSplit:
        return SharedSplit(
            SplitConfig(seed=23, folds=2),
            Split((frozenset(), frozenset()), development=_Table.subjects()),
        )


@dataclass
class _Scorer:
    fail_at: int | None = None
    nonconverged_at: int | None = None
    params: list[Mapping[str, ParamValue]] = field(default_factory=list)
    observed: list[Holdout] = field(default_factory=list)

    def score_folds(
        self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
    ) -> tuple[ValidationScores, ...]:
        if len(self.params) == self.fail_at:
            msg = "syntetické přerušení"
            raise RuntimeError(msg)
        self.params.append(params)
        self.observed.extend(folds)
        scores = []
        for fold in folds:
            positives = [n.note_id for n in fold.test if n.labels]
            negatives = [n.note_id for n in fold.test if not n.labels]
            # Ručně: větev A 2*9/(10+20)=0,6; B 2*8/(10+10)=0,8.
            top = set(
                positives[:9] + negatives[:11]
                if params["branch"] == "r1/logreg"
                else positives[:8] + negatives[:2]
            )
            scores.append(
                ValidationScores(
                    dict.fromkeys(
                        Diagnosis, tuple(float(n.note_id in top) for n in fold.test)
                    ),
                    converged=len(self.params) != self.nonconverged_at,
                    n_features=1,
                )
            )
        return tuple(scores)


_REFINEMENT = RefinementConfig(random_startup_trials=1, widened={})


def _screening() -> tuple[StageTrialRecord, ...]:
    config = stage_config()
    return tuple(
        replace(_record(i, branch.name, value), params=branch.default_params())
        for i, (branch, value) in enumerate(
            zip(config.branches[:2], (0.9, 0.7), strict=True)
        )
    )


@dataclass
class _Searches:
    search: FixedSearch = field(default_factory=FixedSearch)
    config: StageTuningConfig | None = None

    def create(self, config: StageTuningConfig) -> FixedSearch:
        self.config = config
        # Falešná studie navrhuje výchozí bod jediné povolené větve.
        self.search.enqueued.append((-1, config.branches[0].default_params()))
        return self.search


def test_check_picks_family_with_higher_development_score() -> None:
    """Kontrola otočí pořadí předkola, vítězný bod se do studie přenese jen jednou."""
    progress, scorer, searches = (
        MemoryProgress(),
        _Scorer(nonconverged_at=4),
        _Searches(),
    )
    screening = _screening()
    report = StageRefinement(
        _Table(),
        _Split(),
        StratifiedGroupFolds(),
        searches,
        scorer,
        progress,
        StepClock(),
    ).run(stage_config(), _REFINEMENT, screening, trials=6)
    assert report.study is searches.config
    records = progress.records
    assert tuple(c.value for c in report.checks) == pytest.approx((0.6, 0.8))
    assert report.checks == tuple(records[:2])
    assert report.nonconverged_trials == 1
    assert [r.converged for r in records] == [True, True, True, False, True, True, True]
    assert tuple(r.params for r in records[:2]) == tuple(r.params for r in screening)
    assert [r.value for r in records[:2]] == pytest.approx([0.6, 0.8])
    assert all(r.params["branch"] == "r1/complement_nb" for r in records[2:])
    assert len(records) == 7
    assert report.best.params["branch"] == "r1/complement_nb"
    assert report.best.value == max(r.value for r in records)
    assert scorer.params[:2] == [r.params for r in screening]
    assert len(scorer.params) == len(records)
    assert searches.search.restored[0][0].params == screening[1].params
    assert searches.search.restored[0][1] == pytest.approx(0.8)


@pytest.mark.parametrize("cut", [1, 2, 3, 7])
def test_resume_gives_same_outputs_as_uninterrupted_run(
    tmp_path: Path, cut: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Obnova zachová i nekonstantní historii a vítěze kontroly při běhu TPE."""

    class MonotonicScorer(_Scorer):
        def score_folds(
            self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
        ) -> tuple[ValidationScores, ...]:
            scored = super().score_folds(params, folds)
            c = params.get("logreg.C")
            value = 0.1 + 0.8 * (log10(c) + 3) / 6 if isinstance(c, float) else 0.0
            return tuple(
                replace(
                    score, scores=dict.fromkeys(Diagnosis, (value,) * len(fold.test))
                )
                for score, fold in zip(scored, folds, strict=True)
            )

    def metrics(
        notes: Sequence[InputNote], scores: ValidationScores
    ) -> dict[Diagnosis, MaxF1]:
        del notes
        return {d: MaxF1(scores.scores[d][0], 0.5) for d in Diagnosis}

    # Test orchestrace používá analytický cíl v celé log škále C; empirické max-F1
    # má schody a nedokázalo by rozlišit libovolné dvě obnovené hodnoty.
    monkeypatch.setattr("mediparse.application.tuning_trial.fold_max_f1", metrics)
    screening = list(_screening())
    screening[0] = replace(
        screening[0], params=dict(screening[0].params) | {"logreg.C": 0.001}
    )
    checkpoints = [
        ParquetStageCheckpoint(
            tmp_path / name / "parts",
            "key",
            tmp_path / name / "trials",
            tmp_path / name / "predictions",
        )
        for name in ("whole", "resumed")
    ]
    baseline = MemoryProgress()
    StageRefinement(
        _Table(),
        _Split(),
        StratifiedGroupFolds(),
        OptunaStageSearch,
        MonotonicScorer(),
        baseline,
        StepClock(),
    ).run(stage_config(), _REFINEMENT, screening, trials=12)
    records = baseline.records
    winner = best_trial(records[:2])
    assert records[0].value != records[1].value
    assert records[2].value > winner.value
    if cut >= 3:
        assert any(r.value != winner.value for r in records[2:cut])
    if cut == 7:
        assert cut > 1 + 1 + _REFINEMENT.random_startup_trials
    for i, progress in enumerate(checkpoints):
        if i:
            with pytest.raises(RuntimeError, match="přerušení"):
                StageRefinement(
                    _Table(),
                    _Split(),
                    StratifiedGroupFolds(),
                    OptunaStageSearch,
                    MonotonicScorer(fail_at=cut),
                    progress,
                    StepClock(),
                ).run(stage_config(), _REFINEMENT, screening, trials=12)
            assert len(progress.completed()) == cut
        StageRefinement(
            _Table(),
            _Split(),
            StratifiedGroupFolds(),
            OptunaStageSearch,
            MonotonicScorer(),
            progress,
            StepClock(),
        ).run(stage_config(), _REFINEMENT, screening, trials=12)
    assert checkpoints[0].results.read_bytes() == checkpoints[1].results.read_bytes()
    assert (
        checkpoints[0].predictions.read_bytes()
        == checkpoints[1].predictions.read_bytes()
    )


def test_folds_cover_whole_development_part() -> None:
    """Validace je rozkladem všech zpráv vývojové části bez podvzorkování."""
    scorer = _Scorer()
    report = StageRefinement(
        _Table(),
        _Split(),
        StratifiedGroupFolds(),
        OptunaStageSearch,
        scorer,
        MemoryProgress(),
        StepClock(),
    ).run(stage_config(subsample_parts=3), _REFINEMENT, _screening(), trials=6)
    validations = [n.note_id for f in scorer.observed[:2] for n in f.test]
    expected = {n.note_id for n in _Table().read_patients(_Table.subjects())}
    assert len(validations) == len(set(validations)) == report.development_notes == 100
    assert set(validations) == expected
