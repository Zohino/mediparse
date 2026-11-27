"""Mapování labelů adaptéru mezi doménou a numerickým klasifikátorem."""

from __future__ import annotations

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
    notes = corpus()[:4]
    scores = np.tile(np.arange(5), (4, 1))
    classifier = Mock()
    classifier.model_ = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
    classifier.train_epochs.return_value = iter([
        EpochReport(1, 2.0, 0.5, 0, float("nan"), scores)
    ])
    factory = Mock(return_value=classifier)
    monkeypatch.setattr(hf_pilot_trainer, "HfWindowClassifier", factory)
    config = pilot_config()
    (report,) = HfPilotTrainer(7).train(
        config.runs[0], config.training, notes, notes, [n.text for n in notes]
    )
    args = classifier.train_epochs.call_args.args
    np.testing.assert_array_equal(
        args[1], [[n.has_label(d) for d in Diagnosis] for n in notes]
    )
    np.testing.assert_array_equal(args[3], args[1])
    assert args[4] == [n.text for n in notes]
    assert factory.call_args.kwargs["device"] == config.runs[0].encoder.device
    for j, diagnosis in enumerate(Diagnosis):
        assert report.scores.scores[diagnosis] == (float(j),) * 4
    assert report.scores.converged is False
    assert report.scores.n_features == 16
