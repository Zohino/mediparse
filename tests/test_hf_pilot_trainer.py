"""Mapování labelů adaptéru mezi doménou a numerickým klasifikátorem."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from mediparse.domain.diagnosis import Diagnosis
from mediparse.infrastructure import hf_pilot_trainer
from mediparse.infrastructure.hf_classifier import EpochReport
from mediparse.infrastructure.hf_pilot_trainer import HfPilotTrainer
from tests.pilot_support import corpus, pilot_config

if TYPE_CHECKING:
    import pytest


def test_pilot_trainer_maps_diagnoses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pořadí diagnóz řídí labely i skóre a NaN stop BCE znamená nekonvergenci."""
    fit, stop, validation = corpus()[:4], corpus()[4:8], corpus()[8:12]
    scores = np.tile(np.arange(5), (4, 1))
    classifier = Mock()
    classifier.model_ = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
    classifier.train_epochs.return_value = iter([
        EpochReport(2, 3.5, 0.75, 1024, float("nan"), scores)
    ])
    factory = Mock(return_value=classifier)
    monkeypatch.setattr(hf_pilot_trainer, "HfWindowClassifier", factory)
    config = pilot_config()
    run = config.runs[0].model_copy(
        update={
            "aggregation": "mean",
            "freeze_embeddings": False,
            "encoder": config.runs[0].encoder.model_copy(update={"device": "cuda"}),
        }
    )
    training = config.training.model_copy(
        update={"weight_decay": 0.03, "warmup_ratio": 0.2}
    )
    (report,) = HfPilotTrainer(7).train(
        run, training, fit, stop, [n.text for n in validation]
    )
    args = classifier.train_epochs.call_args.args
    np.testing.assert_array_equal(
        args[1], [[n.has_label(d) for d in Diagnosis] for n in fit]
    )
    np.testing.assert_array_equal(
        args[3], [[n.has_label(d) for d in Diagnosis] for n in stop]
    )
    assert args[0] == [n.text for n in fit]
    assert args[2] == [n.text for n in stop]
    assert args[4] == [n.text for n in validation]
    assert factory.call_args.kwargs == {
        "encoder": run.encoder,
        "aggregation": run.aggregation,
        "freeze_embeddings": run.freeze_embeddings,
        "random_state": 7,
        "device": run.encoder.device,
        **training.model_dump(),
    }
    assert (
        report.epoch,
        report.seconds,
        report.validation_seconds,
        report.peak_memory_bytes,
    ) == (2, 3.5, 0.75, 1024)
    assert math.isnan(report.stop_loss)
    for j, diagnosis in enumerate(Diagnosis):
        assert report.scores.scores[diagnosis] == (float(j),) * 4
    assert report.scores.converged is False
    assert report.scores.n_features == 16
