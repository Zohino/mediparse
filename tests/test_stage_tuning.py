"""Výchozí pokusy, vývojová část a obnova ladění stupně."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Barrier, Event
from typing import TYPE_CHECKING

import pytest

from mediparse.application.stage_tuning import StageTuning
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import strata
from mediparse.domain.stage_tuning import StageFoldResult, StageTrialRecord
from mediparse.domain.tuning import MaxF1, Trial, ValidationScores
from mediparse.infrastructure.optuna_search import OptunaStageSearch
from mediparse.infrastructure.parquet_file import read_table
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.stage_trials_table import (
    TRIALS_SCHEMA,
    ParquetStageCheckpoint,
)
from tests.screening_support import DEVELOPMENT, SHARED, notes
from tests.stage_support import (
    CapturingScorer,
    FixedSearch,
    MemoryProgress,
    StepClock,
    stage_config,
    transformer_stage_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from mediparse.application.ports.experiment import TrialScorer
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.split import Holdout, SharedSplit
    from mediparse.domain.tuning import ParamValue


@dataclass(frozen=True)
class _Table:
    """Syntetická tabulka s více zprávami pacienta."""

    @staticmethod
    def subjects() -> frozenset[int]:
        return frozenset(n.subject_id for n in notes())

    @staticmethod
    def read() -> tuple[InputNote, ...]:
        return notes(visits=30)

    def read_patients(self, subjects: frozenset[int]) -> tuple[InputNote, ...]:
        return tuple(n for n in self.read() if n.subject_id in subjects)


@dataclass(frozen=True)
class _SplitSource:
    """Pevný split syntetických pacientů."""

    @staticmethod
    def read() -> SharedSplit:
        return SHARED


_DEVELOPMENT_NOTES = _Table().read_patients(DEVELOPMENT)


@pytest.mark.parametrize(
    ("parts", "expected_patients"),
    [
        (1, DEVELOPMENT),
        (
            3,
            StratifiedGroupFolds().partition(
                [n.subject_id for n in _DEVELOPMENT_NOTES],
                strata(_DEVELOPMENT_NOTES, 3),
                3,
                7,
            )[0],
        ),
    ],
)
def test_defaults_and_development_sample(
    parts: int, expected_patients: frozenset[int]
) -> None:
    """Pouze vývojoví pacienti a všechny jejich zprávy vstupují do disjunktních foldů."""
    config = stage_config(subsample_parts=parts)
    samples = []
    for _ in range(2):
        scorer, progress = CapturingScorer(), MemoryProgress()
        search = OptunaStageSearch.create(config)
        report = StageTuning(
            _Table(),
            _SplitSource(),
            StratifiedGroupFolds(),
            search,
            scorer,
            progress,
            StepClock(),
        ).run(config)
        assert progress.finished
        sample = {n.note_id for f in scorer.observed[:2] for n in f.test}
        patients = {n.subject_id for f in scorer.observed[:2] for n in f.test}
        assert patients == expected_patients
        assert report.sample_patients == len(expected_patients)
        assert sample == {
            n.note_id for n in _Table().read_patients(frozenset(patients))
        }
        if parts == 3:
            assert report.sample_notes < report.development_notes
        for fold in scorer.observed:
            assert {n.subject_id for n in (*fold.train, *fold.test)} <= patients
            assert {n.subject_id for n in fold.train}.isdisjoint(
                n.subject_id for n in fold.test
            )
        assert report.sample_notes == len(sample)
        assert report.best.value == progress.records[report.best.number].value
        samples.append(sample)
    assert samples[0] == samples[1]


def test_fold_features_reach_results() -> None:
    """Počet příznaků každého foldu projde až do všech buněk diagnóz."""
    config = stage_config()
    scorer, progress = CapturingScorer(), MemoryProgress()
    StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        OptunaStageSearch.create(config),
        scorer,
        progress,
        StepClock(),
    ).run(config)
    assert progress.results
    assert all(result.n_features == 10 + result.fold for result in progress.results)


@pytest.mark.parametrize("cut", [3, 5])
def test_interrupted_run_resumes_to_identical_outputs(tmp_path: Path, cut: int) -> None:
    """Pád uvnitř i po výchozích pokusech nemění jediný bajt obou výstupů."""
    config = stage_config()
    checkpoints = [
        ParquetStageCheckpoint(
            tmp_path / name / "parts",
            "key",
            tmp_path / name / "trials.parquet",
            tmp_path / name / "predictions.parquet",
        )
        for name in ("whole", "resumed")
    ]
    for i, progress in enumerate(checkpoints):
        if i:
            with pytest.raises(RuntimeError, match="přerušení"):
                StageTuning(
                    _Table(),
                    _SplitSource(),
                    StratifiedGroupFolds(),
                    OptunaStageSearch.create(config),
                    CapturingScorer(fail_at=cut * 2),
                    progress,
                    StepClock(),
                ).run(config)
            records = progress.completed()
            assert len(records) == cut
            with pytest.raises(InvalidInputError, match="patří jinému běhu"):
                replace(progress, key="foreign").completed()
        StageTuning(
            _Table(),
            _SplitSource(),
            StratifiedGroupFolds(),
            OptunaStageSearch.create(config),
            CapturingScorer(),
            progress,
            StepClock(),
        ).run(config)
        assert not progress.directory.exists()
    assert checkpoints[0].results.read_bytes() == checkpoints[1].results.read_bytes()
    assert (
        checkpoints[0].predictions.read_bytes()
        == checkpoints[1].predictions.read_bytes()
    )


@pytest.mark.parametrize("cut", [0, 3, 5])
def test_enqueue_protocol_and_restore(cut: int) -> None:
    """Výchozí body se vkládají jen před čtyřmi příslušnými ask, při obnově jen zbytek."""
    config = stage_config()
    initial = MemoryProgress()
    StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        FixedSearch(),
        CapturingScorer(),
        initial,
        StepClock(),
    ).run(config)
    records = initial.records[:cut]
    progress, search = MemoryProgress(records=list(records)), FixedSearch()
    if cut >= 4:
        search.enqueued.append((-1, config.branches[-1].default_params()))
    StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        search,
        CapturingScorer(),
        progress,
        StepClock(),
    ).run(config)
    assert search.restored == [(Trial(r.number, r.params), r.value) for r in records]
    assert [(n, p) for n, p in search.enqueued if n >= 0] == [
        (n, branch.default_params())
        for n, branch in enumerate(config.branches)
        if n >= cut
    ]
    assert search.asked == 6
    assert [r.default for r in progress.records] == [True] * 4 + [False] * 2
    assert [r.value for r in progress.records[cut:]] == search.values


def test_checkpoint_round_trip_record_and_fold_results(tmp_path: Path) -> None:
    """Souhrn drží uložený cíl a nekonvergenci libovolného foldu."""
    progress = ParquetStageCheckpoint(
        tmp_path / "parts", "key", tmp_path / "trials", tmp_path / "predictions"
    )
    record = StageTrialRecord(
        0,
        {"word.min_df": 1, "branch": "word/logreg", "logreg.C": 1.0},
        default=True,
        value=0.8,
        converged=False,
        seconds=2.5,
    )
    result = StageFoldResult(
        fold=0,
        diagnosis=Diagnosis.CKD,
        note_ids=("synthetic-1", "synthetic-2"),
        labels=(True, False),
        scores=(0.9, 0.1),
        max_f1=MaxF1(1.0, 0.9),
        n_features=17,
        converged=False,
    )
    progress.write(record, [replace(result, converged=True), replace(result, fold=1)])
    assert progress.completed() == (record,)
    progress.finish()
    assert not progress.directory.exists()
    table = read_table(progress.results, TRIALS_SCHEMA)
    assert table["n_features"].to_pylist() == [17] * table.num_rows
    assert (
        table["params"].to_pylist()
        == ['{"branch": "word/logreg", "logreg.C": 1.0, "word.min_df": 1}'] * 2
    )


def test_nonconvergence_counts_each_affected_trial_once() -> None:
    """Nekonvergence jednoho foldu pokusů 1 a 4 se promítne do obou záznamů."""

    class Nonconverged(CapturingScorer):
        def score(
            self, params: Mapping[str, ParamValue], fold: Holdout
        ) -> ValidationScores:
            scored = super().score(params, fold)
            number = (len(self.observed) - 1) // 2
            return replace(scored, converged=not (number in {1, 4} and fold.fold == 0))

    progress = MemoryProgress()
    report = StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        FixedSearch(),
        Nonconverged(),
        progress,
        StepClock(),
    ).run(stage_config())
    assert [r.converged for r in progress.records] == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert report.nonconverged_trials == 2


class _LoggedSearch(FixedSearch):
    """Studie zaznamenávající otevřené pokusy a pořadí volání."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.opened = 0
        self.maximum = 0

    def enqueue(self, params: Mapping[str, ParamValue]) -> None:
        self.log.append("enqueue")
        super().enqueue(params)

    def ask(self) -> Trial:
        self.log.append("ask")
        self.opened += 1
        self.maximum = max(self.maximum, self.opened)
        return super().ask()

    def complete(self, trial: Trial, value: float) -> None:
        self.log.append("complete")
        self.opened -= 1
        super().complete(trial, value)


class _LoggedProgress(MemoryProgress):
    """Checkpoint s událostí zápisu ve společném logu."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.b_written = Event()

    def write(
        self, record: StageTrialRecord, results: Sequence[StageFoldResult]
    ) -> None:
        self.log.append("write")
        super().write(record, results)
        if record.params["branch"] == "b/transformer":
            self.b_written.set()


def _tuning(
    search: FixedSearch, scorer: TrialScorer, progress: MemoryProgress
) -> StageTuning:
    return StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        search,
        scorer,
        progress,
        StepClock(),
    )


def test_sequential_call_order_is_unchanged() -> None:
    """Čtyři výchozí a dva návrhy zachovají původní pořadí volání."""
    log: list[str] = []
    _tuning(_LoggedSearch(log), CapturingScorer(), _LoggedProgress(log)).run(
        stage_config()
    )
    assert (
        log
        == ["enqueue", "ask", "complete", "write"] * 4
        + ["ask", "complete", "write"] * 2
    )


def test_concurrent_trials_run_simultaneously() -> None:
    """Bariéra vyžaduje skutečný souběh tří pokusů v obou vlnách."""
    barrier = Barrier(3, timeout=10)

    class Scorer(CapturingScorer):
        def score_folds(
            self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
        ) -> tuple[ValidationScores, ...]:
            barrier.wait()
            return super().score_folds(params, folds)

    search = _LoggedSearch([])
    progress = MemoryProgress()
    _tuning(search, Scorer(), progress).run(
        transformer_stage_config(concurrent_trials=3)
    )
    assert len(progress.records) == 6
    assert search.maximum == 3


def test_out_of_order_completion_keeps_protocol() -> None:
    """Dokončení B před A neztratí výchozí body ani souvislé číslování."""
    progress = _LoggedProgress([])

    class Scorer(CapturingScorer):
        def score_folds(
            self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
        ) -> tuple[ValidationScores, ...]:
            if params["branch"] == "a/transformer":
                assert progress.b_written.wait(10)
            return super().score_folds(params, folds)

    config = transformer_stage_config(concurrent_trials=2)
    _tuning(_LoggedSearch([]), Scorer(), progress).run(config)
    assert [r.number for r in progress.records] == list(range(6))
    assert progress.records[0].params["branch"] == "b/transformer"
    for branch in config.branches:
        assert [
            r.params
            for r in progress.records
            if r.default and r.params["branch"] == branch.name
        ] == [branch.default_params()]


@pytest.mark.parametrize("failed", [1, 2])
def test_concurrent_failure_drains_running_trials(failed: int) -> None:
    """Selhání zastaví návrhy a dovolí zapsat zbylé dva běžící pokusy."""
    barrier = Barrier(3, timeout=10)
    failure = Event()
    log: list[str] = []
    config = transformer_stage_config(concurrent_trials=3)
    family = config.model_dump(mode="json")["families"][0]
    family["space"]["params"][0] = {
        "kind": "float",
        "name": "lr",
        "low": 1e-5,
        "high": 1e-4,
        "log": True,
    }
    config = transformer_stage_config(concurrent_trials=3, families=[family])

    class Search(_LoggedSearch):
        def ask(self) -> Trial:
            trial = super().ask()
            return (
                replace(trial, params=dict(trial.params) | {"transformer.lr": 1e-5})
                if trial.number == 2
                else trial
            )

    search = Search(log)
    failed_params = config.branches[1].default_params()
    if failed == 2:
        failed_params |= {"transformer.lr": 1e-5}

    class Scorer(CapturingScorer):
        def score_folds(
            self, params: Mapping[str, ParamValue], folds: Sequence[Holdout]
        ) -> tuple[ValidationScores, ...]:
            barrier.wait()
            if params == failed_params:
                log.append("failure")
                failure.set()
                msg = "přerušení"
                raise RuntimeError(msg)
            assert failure.wait(10)
            return super().score_folds(params, folds)

    progress = MemoryProgress()
    with pytest.raises(RuntimeError, match=r"^přerušení$"):
        _tuning(search, Scorer(), progress).run(config)
    # První vlna: A výchozí, B výchozí, B nevýchozí; jeden selže, dva se zapíšou.
    assert search.asked == 3
    assert len(progress.records) == 2
    assert "ask" not in log[log.index("failure") + 1 :]
    assert not progress.finished


def test_resume_restores_missing_defaults() -> None:
    """Checkpoint po SIGKILL doplní jen chybějící výchozí bod B."""
    config = transformer_stage_config(concurrent_trials=2)
    a = config.branches[0].default_params()
    progress = MemoryProgress(
        records=[
            StageTrialRecord(
                0, a, default=True, value=0.5, converged=True, seconds=1.0
            ),
            StageTrialRecord(
                1, a, default=False, value=0.4, converged=True, seconds=1.0
            ),
            StageTrialRecord(
                2, a, default=False, value=0.3, converged=True, seconds=1.0
            ),
        ]
    )
    search = FixedSearch()
    _tuning(search, CapturingScorer(), progress).run(config)
    assert search.enqueued == [(3, config.branches[1].default_params())]
    assert sorted(r.number for r in progress.records) == list(range(6))
    for branch in config.branches:
        assert [
            r.params
            for r in progress.records
            if r.default and r.params["branch"] == branch.name
        ] == [branch.default_params()]


def test_concurrent_errors_raise_first_started_and_log_only_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pozdější selhání B nevyhraje nad A a log neobsahuje obsah výjimky."""
    barrier = Barrier(2, timeout=10)
    b_failed = Event()

    class Scorer:
        @staticmethod
        def score_folds(
            params: Mapping[str, ParamValue], folds: Sequence[Holdout]
        ) -> tuple[ValidationScores, ...]:
            del folds
            barrier.wait()
            if params["branch"] == "b/transformer":
                b_failed.set()
                msg = "syntetický obsah výjimky B"
                raise ValueError(msg)
            assert b_failed.wait(10)
            msg = "první pokus"
            raise RuntimeError(msg)

    search = FixedSearch()
    with pytest.raises(RuntimeError, match="první pokus"):
        _tuning(search, Scorer(), MemoryProgress()).run(
            transformer_stage_config(concurrent_trials=2)
        )
    assert search.asked == 2
    assert caplog.messages == ["Pokus 1 selhal: ValueError"]
