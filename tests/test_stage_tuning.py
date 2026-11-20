"""Výchozí pokusy, vývojová část a obnova ladění stupně."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from tests.stage_support import StepClock, stage_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

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


@dataclass
class _Scorer:
    """Deterministická skóre a volitelné přerušení."""

    fail_at: int | None = None
    observed: list[Holdout] = field(default_factory=list)

    def score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        del params
        if len(self.observed) == self.fail_at:
            msg = "syntetické přerušení"
            raise RuntimeError(msg)
        self.observed.append(fold)
        return ValidationScores(
            {
                d: tuple(
                    float(n.has_label(d)) if fold.fold == 0 else 0.0 for n in fold.test
                )
                for d in Diagnosis
            },
            converged=True,
            features=len(fold.train),
        )


@dataclass
class _Progress:
    """Paměťový checkpoint pro ověření protokolu."""

    records: list[StageTrialRecord] = field(default_factory=list)
    finished: bool = False

    def completed(self) -> tuple[StageTrialRecord, ...]:
        return tuple(self.records)

    def write(
        self, record: StageTrialRecord, results: Sequence[StageFoldResult]
    ) -> None:
        del results
        self.records.append(record)

    def finish(self) -> None:
        self.finished = True


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
        scorer, progress = _Scorer(), _Progress()
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
                    _Scorer(fail_at=cut * 2),
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
            _Scorer(),
            progress,
            StepClock(),
        ).run(config)
        assert not progress.directory.exists()
    assert checkpoints[0].results.read_bytes() == checkpoints[1].results.read_bytes()
    assert (
        checkpoints[0].predictions.read_bytes()
        == checkpoints[1].predictions.read_bytes()
    )


@dataclass
class _Search:
    asked: int = 0
    enqueued: list[tuple[int, Mapping[str, ParamValue]]] = field(default_factory=list)
    restored: list[tuple[Trial, float]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def restore(self, trials: Sequence[tuple[Trial, float]]) -> None:
        self.restored.extend(trials)
        self.asked = len(trials)

    def enqueue(self, params: Mapping[str, ParamValue]) -> None:
        self.enqueued.append((self.asked, params))

    def ask(self) -> Trial:
        trial = Trial(self.asked, self.enqueued[-1][1])
        self.asked += 1
        return trial

    @staticmethod
    def should_prune(trial: Trial, value: float, step: int) -> bool:
        del trial, value, step
        return False

    def complete(self, trial: Trial, value: float) -> None:
        del trial
        self.values.append(value)

    @staticmethod
    def prune(trial: Trial) -> None:
        del trial
        pytest.fail("Stupeň neprořezává.")


@pytest.mark.parametrize("cut", [0, 3, 5])
def test_enqueue_protocol_and_restore(cut: int) -> None:
    """Výchozí body se vkládají jen před čtyřmi příslušnými ask, při obnově jen zbytek."""
    config = stage_config()
    initial = _Progress()
    StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        _Search(),
        _Scorer(),
        initial,
        StepClock(),
    ).run(config)
    records = initial.records[:cut]
    progress, search = _Progress(records=list(records)), _Search()
    if cut >= 4:
        search.enqueued.append((-1, config.branches[-1].default_params()))
    StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        search,
        _Scorer(),
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
        converged=False,
    )
    progress.write(record, [replace(result, converged=True), replace(result, fold=1)])
    assert progress.completed() == (record,)
    progress.finish()
    assert not progress.directory.exists()
    assert (
        read_table(progress.results, TRIALS_SCHEMA)["params"].to_pylist()
        == ['{"branch": "word/logreg", "logreg.C": 1.0, "word.min_df": 1}'] * 2
    )


def test_stage_rejects_pruned_trial() -> None:
    """Porušení smlouvy hledání bez pruneru nesmí vytvořit dokončený výstup."""

    class PruningSearch(_Search):
        @staticmethod
        def should_prune(trial: Trial, value: float, step: int) -> bool:
            del trial, value, step
            return True

        @staticmethod
        def prune(trial: Trial) -> None:
            del trial

    progress = _Progress()
    with pytest.raises(InvalidInputError, match="neprořezává"):
        StageTuning(
            _Table(),
            _SplitSource(),
            StratifiedGroupFolds(),
            PruningSearch(),
            _Scorer(),
            progress,
            StepClock(),
        ).run(stage_config())
    assert not progress.records
    assert not progress.finished


def test_nonconvergence_counts_each_affected_trial_once() -> None:
    """Nekonvergence jednoho foldu pokusů 1 a 4 se promítne do obou záznamů."""

    class Nonconverged(_Scorer):
        def score(
            self, params: Mapping[str, ParamValue], fold: Holdout
        ) -> ValidationScores:
            scored = super().score(params, fold)
            number = (len(self.observed) - 1) // 2
            return replace(scored, converged=not (number in {1, 4} and fold.fold == 0))

    progress = _Progress()
    report = StageTuning(
        _Table(),
        _SplitSource(),
        StratifiedGroupFolds(),
        _Search(),
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
