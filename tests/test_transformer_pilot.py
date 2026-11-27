"""Oddělení pacientů, shoda předkola a ruční orákulum macro max-F1."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from mediparse.application.patient_cv import development_sample, patient_holdouts
from mediparse.application.stage_tuning import StageTuning
from mediparse.application.transformer_pilot import TransformerPilot
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import patient_fraction
from mediparse.infrastructure.optuna_search import OptunaStageSearch
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds, holdout_mask
from tests.pilot_support import Source, Table, Trainer, corpus, pilot_config
from tests.stage_support import MemoryProgress, StepClock, stage_config
from tests.test_stage_tuning import _Scorer  # ruff: ignore[import-private-name]


def test_records_per_run_and_epoch(caplog: pytest.LogCaptureFixture) -> None:
    """Dva běhy po třech epochách zapisují kumulativně tři a šest řádků."""
    sink = Mock()
    caplog.set_level("INFO")
    report = TransformerPilot(
        Table(), Source(), StratifiedGroupFolds(), Trainer(), sink, StepClock()
    ).run(pilot_config())
    assert [(r.run, r.epoch) for r in report.records] == [
        (name, epoch) for name in ("run0", "run1") for epoch in (1, 2, 3)
    ]
    assert [len(call.args[0]) for call in sink.write.call_args_list] == [3, 6]
    assert report.development_notes == report.development_patients == 1200
    assert report.sample_notes == report.sample_patients == 1200
    assert report.split_sha256 == Source().read().split.sha256()
    assert len(caplog.records) == 2


def test_validation_is_fold_zero_of_stage_sample() -> None:
    """StageTuning a pilot používají stejnou validaci, fit a stop jsou disjunktní."""
    config = pilot_config(parts=3)
    trainer = Trainer()
    partitioner = StratifiedGroupFolds()
    TransformerPilot(Table(), Source(), partitioner, trainer, Mock(), StepClock()).run(
        config
    )
    stage = stage_config(
        seed=config.seed, subsample_parts=config.subsample_parts, folds=config.folds
    )
    scorer = _Scorer()
    StageTuning(
        Table(),
        Source(),
        partitioner,
        OptunaStageSearch.create(stage),
        scorer,
        MemoryProgress(),
        StepClock(),
    ).run(stage)
    sample = development_sample(corpus(), 3, config.seed, partitioner)
    fold0 = patient_holdouts(sample, config.folds, config.seed, partitioner)[0]
    assert [observed[0] for observed in trainer.observed] == list(config.runs)
    for run, (_, fit, stop, validation) in zip(
        config.runs, trainer.observed, strict=True
    ):
        assert (
            list(validation)
            == [n.text for n in scorer.observed[0].test]
            == [n.text for n in fold0.test]
        )
        subset = patient_fraction(fold0.train, run.fraction, config.seed)
        mask = holdout_mask(subset, config.stopping_parts, config.seed)
        assert tuple(stop) == tuple(
            n for n, held in zip(subset, mask, strict=True) if held
        )
        assert tuple(fit) == tuple(
            n for n, held in zip(subset, mask, strict=True) if not held
        )
        fit_patients, stop_patients = (
            {n.subject_id for n in group} for group in (fit, stop)
        )
        assert fit_patients.isdisjoint(stop_patients)
        assert {n.subject_id for n in fold0.test}.isdisjoint(
            fit_patients | stop_patients
        )


def test_macro_f1_from_scores() -> None:
    """P=N=2k: maxima přes prahy jsou 2/3, 1/2, 4/5, 2/3, tedy 0,8."""
    trainer = Trainer()
    report = TransformerPilot(
        Table(), Source(), StratifiedGroupFolds(), trainer, Mock(), StepClock()
    ).run(pilot_config(epochs=1))
    for _, _, _, validation in trainer.observed:
        positives = sum(int(text) % 2 for text in validation)
        negatives = len(validation) - positives
        assert positives == negatives
        assert positives % 2 == 0
        assert positives >= 10
    assert all(r.macro_f1 == pytest.approx(0.8) for r in report.records)


def test_patient_outside_split_refused() -> None:
    """Tabulka s pacientem mimo split musí selhat ještě před tréninkem."""
    extra = replace(corpus()[0], subject_id=90009999, note_id="90009999-DS-0")
    trainer = Trainer()
    with pytest.raises(InvalidInputError):
        TransformerPilot(
            Table((*corpus(), extra)),
            Source(),
            StratifiedGroupFolds(),
            trainer,
            Mock(),
            StepClock(),
        ).run(pilot_config())
    assert not trainer.observed
