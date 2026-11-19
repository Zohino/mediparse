"""Výchozí pokusy, vývojová část a obnova ladění stupně."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import pytest

from mediparse.application.stage_tuning import StageTuning
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.optuna_search import OptunaStageSearch
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.stage_trials_table import ParquetStageCheckpoint
from tests.screening_support import DEVELOPMENT, SHARED, notes
from tests.stage_support import stage_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.input_note import InputNote
    from mediparse.domain.split import Holdout, SharedSplit
    from mediparse.domain.stage_tuning import StageFoldResult, StageTrialRecord
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
class _Clock:
    """Hodiny s konstantním krokem."""

    value: float = 0.0

    def now(self) -> float:
        self.value += 1.0
        return self.value


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


@pytest.mark.parametrize("parts", [1, 3])
def test_defaults_and_development_sample(parts: int) -> None:
    """Pouze vývojoví pacienti a všechny jejich zprávy vstupují do disjunktních foldů."""
    config = stage_config(subsample_parts=parts)
    samples = []
    for _ in range(2):
        scorer, progress = _Scorer(), _Progress()
        search = OptunaStageSearch.create(config.branches, config.seed, 5)
        report = StageTuning(
            _Table(),
            _SplitSource(),
            StratifiedGroupFolds(),
            search,
            scorer,
            progress,
            _Clock(),
        ).run(config)
        assert progress.finished
        assert [r.default for r in progress.records] == [True] * 4 + [False] * 2
        assert [r.params for r in progress.records[:4]] == [
            b.default_params() for b in config.branches
        ]
        sample = {n.note_id for f in scorer.observed[:2] for n in f.test}
        patients = {n.subject_id for f in scorer.observed[:2] for n in f.test}
        assert patients <= DEVELOPMENT
        assert sample == {
            n.note_id for n in _Table().read_patients(frozenset(patients))
        }
        if parts == 1:
            assert patients == DEVELOPMENT
        for fold in scorer.observed:
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
                    OptunaStageSearch.create(config.branches, config.seed, 5),
                    _Scorer(fail_at=cut * 2),
                    progress,
                    _Clock(),
                ).run(config)
            records = progress.completed()
            assert len(records) == cut
            with pytest.raises(InvalidInputError, match="patří jinému běhu"):
                replace(progress, key="foreign").completed()
        StageTuning(
            _Table(),
            _SplitSource(),
            StratifiedGroupFolds(),
            OptunaStageSearch.create(config.branches, config.seed, 5),
            _Scorer(),
            progress,
            _Clock(),
        ).run(config)
        assert not progress.directory.exists()
    assert checkpoints[0].results.read_bytes() == checkpoints[1].results.read_bytes()
    assert (
        checkpoints[0].predictions.read_bytes()
        == checkpoints[1].predictions.read_bytes()
    )
