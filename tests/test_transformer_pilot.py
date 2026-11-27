"""Oddělení pacientů, shoda předkola a ruční orákulum macro max-F1."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from mediparse.application.patient_cv import development_sample, patient_holdouts
from mediparse.application.stage_tuning import StageTuning
from mediparse.application.transformer_pilot import TransformerPilot
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import patient_fraction
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.optuna_search import OptunaStageSearch
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds, holdout_mask
from tests.pilot_support import Source, Table, Trainer, corpus, pilot_config
from tests.stage_support import CapturingScorer, MemoryProgress, StepClock, stage_config

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mediparse.domain.input_note import InputNote
    from mediparse.domain.transformer_pilot import PilotEpoch, PilotRun, PilotTraining


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
    for record, name in zip(caplog.records, ("run0", "run1"), strict=True):
        assert isinstance(record.args, tuple)
        assert record.args[0] == name
    assert report.seconds == pytest.approx(1.0)


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
    scorer = CapturingScorer()
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
    assert len(trainer.observed) == 2
    assert len(report.records) == 2
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


@pytest.mark.parametrize(
    "losses", [(float("nan"), 0.2, 0.5), (float("nan"), float("inf"), -float("inf"))]
)
def test_log_selects_only_finite_stop_loss(
    losses: tuple[float, ...], caplog: pytest.LogCaptureFixture
) -> None:
    """Log vybírá konečné minimum, při úplné divergenci neuvádí F1."""

    def train(
        run: PilotRun,
        training: PilotTraining,
        fit: Sequence[InputNote],
        stop: Sequence[InputNote],
        validation: Sequence[str],
    ) -> Iterator[PilotEpoch]:
        for epoch in Trainer().train(run, training, fit, stop, validation):
            scores = (
                epoch.scores
                if epoch.epoch != 2
                else ValidationScores(
                    {
                        d: tuple(float(int(text) % 2) for text in validation)
                        for d in Diagnosis
                    },
                    converged=True,
                    n_features=16,
                )
            )
            yield replace(epoch, stop_loss=losses[epoch.epoch - 1], scores=scores)

    trainer = Mock()
    trainer.train.side_effect = train
    caplog.set_level("INFO")
    TransformerPilot(
        Table(), Source(), StratifiedGroupFolds(), trainer, Mock(), StepClock()
    ).run(pilot_config())
    assert len(caplog.records) == 2
    for record in caplog.records:
        if losses[1] == pytest.approx(0.2):
            assert isinstance(record.args, tuple)
            assert record.args[2] == pytest.approx(1.0)
        else:
            assert "nekonvergoval" in record.message
            assert "macro-F1" not in record.message
