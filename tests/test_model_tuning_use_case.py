"""Orákulum smyčky a vyloučení pacientů testu i vývoje z vnitřní CV."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from operator import itemgetter
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mediparse.application.model_tuning import ModelTuning
from mediparse.application.patient_cv import patient_holdouts
from mediparse.application.tuning_trial import run_trial
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import SharedSplit, Split, SplitConfig
from mediparse.domain.tuning import Trial, ValidationScores
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from tests.screening_support import DEVELOPMENT, notes
from tests.tuning_support import SVM_PARAMS, matrix_row, tuning_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.application.ports.experiment import PatientPartitioner
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout
    from mediparse.domain.tuning import ParamValue, TuningReport

SHARED = SharedSplit(
    SplitConfig(seed=23, folds=2),
    Split(
        (frozenset(sorted(DEVELOPMENT)[:10]), frozenset(sorted(DEVELOPMENT)[10:])),
        development=frozenset({1, 2}),
    ),
)


@dataclass(frozen=True)
class _Table:
    @staticmethod
    def read() -> tuple[InputNote, ...]:
        return notes(visits=10)


@dataclass(frozen=True)
class _Split:
    @staticmethod
    def read() -> SharedSplit:
        return SHARED


@dataclass
class _Search:
    prune_last: bool = False
    asked: int = 0
    reported: list[tuple[int, float, int]] = field(default_factory=list)
    completed: dict[int, tuple[Mapping[str, ParamValue], float]] = field(
        default_factory=dict
    )
    pruned: list[int] = field(default_factory=list)

    def ask(self) -> Trial:
        trial = Trial(self.asked, SVM_PARAMS | {"c": 0.5 + self.asked})
        self.asked += 1
        return trial

    def should_prune(self, trial: Trial, value: float, step: int) -> bool:
        self.reported.append((trial.number, value, step))
        return trial.number == 1 and step == (1 if self.prune_last else 0)

    def complete(self, trial: Trial, value: float) -> None:
        self.completed[trial.number] = trial.params, value

    def prune(self, trial: Trial) -> None:
        self.pruned.append(trial.number)

    def best(self) -> tuple[Mapping[str, ParamValue], float]:
        return max(self.completed.values(), key=itemgetter(1))


@dataclass(frozen=True)
class _Scorer:
    observed: list[Holdout] = field(default_factory=list)

    @property
    def parameters(self) -> frozenset[str]:
        return frozenset(SVM_PARAMS)

    def score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        self.observed.append(fold)
        perfect = params["c"] in {0.5, 1.5} and fold.fold == 0
        scores = {
            diagnosis: tuple(
                float(note.has_label(diagnosis)) if perfect else 0.0
                for note in fold.test
            )
            for diagnosis in Diagnosis
        }
        return ValidationScores(
            scores, converged=params["c"] in {1.5, 2.5} or fold.fold != 0, features=1
        )


@dataclass(frozen=True)
class _PairedParts:
    seeds: list[Seed] = field(default_factory=list)

    def partition(
        self, subjects: Sequence[int], strata: Sequence[int], parts: int, seed: Seed
    ) -> tuple[frozenset[int], ...]:
        del strata
        self.seeds.append(seed)
        pairs = {
            subject: index // 2 for index, subject in enumerate(sorted(set(subjects)))
        }
        return tuple(
            frozenset(
                subject for subject, pair in pairs.items() if pair % parts == part
            )
            for part in range(parts)
        )


def _run(
    search: _Search,
    scorer: _Scorer,
    partitioner: PatientPartitioner,
    *,
    fold: int = 0,
    seed: int = 7,
) -> TuningReport:
    return ModelTuning(_Table(), _Split(), partitioner, search, scorer).run(
        tuning_config(fold=fold, seed=seed), matrix_row()
    )


def test_trial_loop_uses_budget_running_means_and_counts_nonconvergence_once() -> None:
    """Trial 0 má foldy 1 a 2/3 (průměr 5/6), trial 1 se prořeže, trial 2 má 2/3."""
    search, scorer = _Search(), _Scorer()
    partitioner = _PairedParts()
    report = _run(search, scorer, partitioner)
    assert partitioner.seeds == [tuning_config().seed]
    assert (report.outcome.completed_trials, report.outcome.pruned_trials) == (2, 1)
    assert report.outcome.nonconverged_trials == 1
    assert report.outcome.best == SVM_PARAMS
    assert report.outcome.best_macro_f1 == pytest.approx(5 / 6)
    assert (search.asked, search.pruned, list(search.completed)) == (3, [1], [0, 2])
    assert [(trial, step) for trial, _, step in search.reported] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    assert [value for _, value, _ in search.reported] == pytest.approx([
        1.0,
        1.0,
        2 / 3,
    ])
    assert (report.train_notes, report.train_patients) == (100, 10)
    assert report.split_config == SHARED.config
    assert report.split_sha256 == SHARED.split.sha256()


@given(fold=st.integers(0, 1), seed=st.integers(0, 2**32 - 1))
def test_inner_cv_sees_only_outer_training_patients(fold: int, seed: int) -> None:
    """Pacienti testu a vývoje nevstoupí ani do fitu, ani do validace; validace pokryjí trénink jednou."""
    scorer = _Scorer()
    _run(_Search(), scorer, StratifiedGroupFolds(), fold=fold, seed=seed)
    holdout = SHARED.split.holdout(notes(visits=10), fold)
    expected = {note.subject_id for note in holdout.train}
    forbidden = SHARED.split.folds[fold] | SHARED.split.development
    seen = set()
    validated: Counter[int] = Counter()
    for inner in scorer.observed[:2]:
        train = {note.subject_id for note in inner.train}
        validation = {note.subject_id for note in inner.test}
        assert train.isdisjoint(validation)
        assert (train | validation).isdisjoint(forbidden)
        seen |= train | validation
        validated.update(validation)
    assert seen == expected
    assert validated == Counter(dict.fromkeys(expected, 1))
    for inner in scorer.observed:
        assert {note.subject_id for note in (*inner.train, *inner.test)} <= expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"row_id": "other"}, "neodpovídá row_id"),
        (
            {
                "space": {
                    "params": [{"kind": "categorical", "name": "C", "choices": [1.0]}]
                }
            },
            "Jména prostoru",
        ),
    ],
)
def test_mismatched_identity_or_parameters_are_refused(
    changes: dict[str, object],
    message: str,
) -> None:
    """Cizí řádek a překlep C se odmítnou před prvním pokusem."""
    search = _Search()
    with pytest.raises(InvalidInputError, match=message):
        ModelTuning(_Table(), _Split(), _PairedParts(), search, _Scorer()).run(
            tuning_config(**changes), matrix_row()
        )
    assert search.asked == 0


def test_scorer_failure_aborts_without_counting_failed_trial() -> None:
    """Chyba scoreru propadne, další ask ani uzavření FAIL se neprovádí."""

    @dataclass(frozen=True)
    class BrokenScorer:
        parameters: frozenset[str] = frozenset(SVM_PARAMS)

        @staticmethod
        def score(params: Mapping[str, ParamValue], fold: Holdout) -> ValidationScores:
            del params, fold
            msg = "syntetické selhání scoreru"
            raise RuntimeError(msg)

    search = _Search()
    with pytest.raises(RuntimeError, match="syntetické selhání"):
        ModelTuning(_Table(), _Split(), _PairedParts(), search, BrokenScorer()).run(
            tuning_config(), matrix_row()
        )
    assert search.asked == 1
    assert not search.completed
    assert not search.pruned


def test_last_fold_always_completes_trial() -> None:
    """Prořezání po posledním foldu už nešetří výpočet, search se na něj nezeptá."""
    search = _Search(prune_last=True)
    report = _run(search, _Scorer(), _PairedParts())
    assert (report.outcome.completed_trials, report.outcome.pruned_trials) == (3, 0)
    assert all(step == 0 for _, _, step in search.reported)


def test_completed_trial_returns_fold_metrics_and_nonconvergence() -> None:
    """Dva dokončené foldy dávají cíl 5/6 a zachovají nekonvergenci."""
    search = _Search()
    folds = patient_holdouts(notes(visits=10), 2, 7, _PairedParts())
    result = run_trial(search, _Scorer(), Trial(0, SVM_PARAMS), folds)
    assert not result.pruned
    assert not result.converged
    assert result.value == pytest.approx(5 / 6)
    assert len(result.f1) == 2
    assert all(list(metrics) == list(Diagnosis) for metrics in result.f1)
    assert search.completed[0][1] == result.value


def test_pruned_trial_returns_only_evaluated_fold() -> None:
    """Prořezaný pokus vrátí jen první fold s cílem 1 a nekonvergencí."""
    search = _Search()
    folds = patient_holdouts(notes(visits=10), 2, 7, _PairedParts())
    result = run_trial(search, _Scorer(), Trial(1, SVM_PARAMS), folds)
    assert result.pruned
    assert not result.converged
    assert len(result.f1) == 1
    assert result.value == pytest.approx(1.0)
    assert search.pruned == [1]


def test_pruned_nonconverged_trial_counts_in_model_outcome() -> None:
    """Nekonvergence prořezaného pokusu se započítá stejně jako dokončená."""

    class Nonconverged(_Scorer):
        def score(
            self, params: Mapping[str, ParamValue], fold: Holdout
        ) -> ValidationScores:
            return replace(super().score(params, fold), converged=False)

    report = _run(_Search(), Nonconverged(), _PairedParts())
    assert report.outcome.pruned_trials == 1
    assert report.outcome.nonconverged_trials == 3
