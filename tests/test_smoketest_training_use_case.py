"""Use case tréninku: labely zvolené diagnózy, trénink jen na train, metriky jen na test."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.smoketest_training import SmoketestTraining
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import InputNote
from mediparse.domain.smoketest_training import (
    BinaryMetrics,
    Holdout,
    SubjectLeakError,
    TrainingConfig,
    TrainingReport,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.smoketest_training import TextClassifier

CONFIG = TrainingConfig(diagnosis=Diagnosis.CKD, seed=0, folds=2, regularization=1.0)
METRICS = BinaryMetrics(0.5, 0.5, 0.5, 0.5)
NOTES = (
    InputNote("a", 1, "en", "text a", frozenset({Diagnosis.CKD})),
    InputNote("b", 2, "en", "text b", frozenset({Diagnosis.DIABETES})),
    InputNote("c", 3, "en", "text c", frozenset({Diagnosis.CKD})),
    InputNote("d", 4, "en", "text d", frozenset()),
)


@dataclass
class _Recorder:
    calls: dict[str, object] = field(default_factory=dict)


@dataclass
class _FakeTable:
    notes: Sequence[InputNote]

    def read(self) -> tuple[InputNote, ...]:
        return tuple(self.notes)


@dataclass
class _FakeSplitter:
    holdout: Holdout
    recorder: _Recorder

    def split(self, labels: Sequence[bool], groups: Sequence[int]) -> Holdout:
        self.recorder.calls["split"] = (tuple(labels), tuple(groups))
        return self.holdout


@dataclass
class _FakeClassifier:
    answer: tuple[bool, ...]
    recorder: _Recorder

    def predict(self, texts: Sequence[str]) -> tuple[bool, ...]:
        self.recorder.calls["predict"] = tuple(texts)
        return self.answer


@dataclass
class _FakeTrainer:
    recorder: _Recorder

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> _FakeClassifier:
        self.recorder.calls["fit"] = (tuple(texts), tuple(labels))
        return _FakeClassifier((True, False), self.recorder)


@dataclass
class _FakeScorer:
    recorder: _Recorder

    def score(self, truth: Sequence[bool], predicted: Sequence[bool]) -> BinaryMetrics:
        self.recorder.calls["score"] = (tuple(truth), tuple(predicted))
        return METRICS


@dataclass
class _FakeStore:
    recorder: _Recorder

    def save(self, classifier: TextClassifier) -> None:
        self.recorder.calls["save"] = classifier


@dataclass
class _FakeSink:
    recorder: _Recorder

    def write(self, report: TrainingReport) -> None:
        self.recorder.calls["report"] = report


def _use_case(
    notes: Sequence[InputNote], holdout: Holdout, recorder: _Recorder
) -> SmoketestTraining:
    return SmoketestTraining(
        table=_FakeTable(notes),
        splitter=_FakeSplitter(holdout, recorder),
        trainer=_FakeTrainer(recorder),
        scorer=_FakeScorer(recorder),
        store=_FakeStore(recorder),
        report=_FakeSink(recorder),
    )


def test_trains_on_train_and_scores_on_test() -> None:
    """Labely jsou zvolené diagnózy; trénink vidí jen train, skóre jen test."""
    recorder = _Recorder()

    report = _use_case(NOTES, Holdout(train=(0, 1), test=(2, 3)), recorder).run(CONFIG)

    assert recorder.calls["split"] == ((True, False, True, False), (1, 2, 3, 4))
    assert recorder.calls["fit"] == (("text a", "text b"), (True, False))
    assert recorder.calls["predict"] == ("text c", "text d")
    assert recorder.calls["score"] == ((True, False), (True, False))
    assert isinstance(recorder.calls["save"], _FakeClassifier)
    assert recorder.calls["report"] == report
    assert report == TrainingReport(Diagnosis.CKD, 2, 2, 1, METRICS)


def test_shared_subject_stops_before_saving() -> None:
    """Split sdílející pacienta vyhodí SubjectLeakError a nic se neuloží."""
    notes = (*NOTES[:3], InputNote("e", 1, "en", "text e", frozenset()))
    recorder = _Recorder()

    with pytest.raises(SubjectLeakError):
        _use_case(notes, Holdout(train=(0, 1), test=(2, 3)), recorder).run(CONFIG)

    assert "fit" not in recorder.calls
    assert "save" not in recorder.calls
    assert "report" not in recorder.calls


def test_class_with_too_few_patients_stops_before_splitting() -> None:
    """Diagnóza bez dost pacientů v každé třídě skončí InvalidInputError a nic se nevolá."""
    config = TrainingConfig(
        diagnosis=Diagnosis.AKI, seed=0, folds=2, regularization=1.0
    )
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="aki"):
        _use_case(NOTES, Holdout(train=(0, 1), test=(2, 3)), recorder).run(config)

    assert recorder.calls == {}
