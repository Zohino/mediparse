"""Use case tréninku: labely zvolené diagnózy, trénink jen na train, predikce jen testu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.smoketest_training import SmoketestTraining
from mediparse.domain.evaluation import Classification, Prediction
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import InputNote
from mediparse.domain.smoketest_training import (
    Holdout,
    SubjectLeakError,
    TrainingConfig,
    TrainingReport,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports import TextClassifier

CONFIG = TrainingConfig(
    row_id="row", diagnosis=Diagnosis.CKD, seed=0, folds=2, regularization=1.0
)
NOTES = (
    InputNote("a", 1, "en", "text a", frozenset({Diagnosis.CKD})),
    InputNote("b", 2, "en", "text b", frozenset({Diagnosis.DIABETES})),
    InputNote("c", 3, "en", "text c", frozenset({Diagnosis.CKD})),
    InputNote("d", 4, "en", "text d", frozenset()),
)


ANSWER = (
    Classification(predicted=True, score=1.5),
    Classification(predicted=False, score=-0.5),
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
    answer: tuple[Classification, ...]
    recorder: _Recorder

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        self.recorder.calls["classify"] = tuple(texts)
        return self.answer


@dataclass
class _FakeTrainer:
    recorder: _Recorder

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> _FakeClassifier:
        self.recorder.calls["fit"] = (tuple(texts), tuple(labels))
        return _FakeClassifier(ANSWER, self.recorder)


@dataclass
class _FakeStore:
    recorder: _Recorder

    def save(self, classifier: TextClassifier) -> None:
        self.recorder.calls["save"] = classifier


@dataclass
class _FakeSink:
    recorder: _Recorder

    def write(self, predictions: Sequence[Prediction]) -> None:
        self.recorder.calls["predictions"] = tuple(predictions)


def _use_case(
    notes: Sequence[InputNote], holdout: Holdout, recorder: _Recorder
) -> SmoketestTraining:
    return SmoketestTraining(
        table=_FakeTable(notes),
        splitter=_FakeSplitter(holdout, recorder),
        trainer=_FakeTrainer(recorder),
        store=_FakeStore(recorder),
        predictions=_FakeSink(recorder),
    )


def test_trains_on_train_and_predicts_test() -> None:
    """Labely jsou zvolené diagnózy; trénink vidí jen train, predikce jen test."""
    recorder = _Recorder()

    report = _use_case(NOTES, Holdout(train=(0, 1), test=(2, 3)), recorder).run(CONFIG)

    assert recorder.calls["split"] == ((True, False, True, False), (1, 2, 3, 4))
    assert recorder.calls["fit"] == (("text a", "text b"), (True, False))
    assert recorder.calls["classify"] == ("text c", "text d")
    assert isinstance(recorder.calls["save"], _FakeClassifier)
    assert recorder.calls["predictions"] == (
        Prediction(
            "row", 0, "c", 3, Diagnosis.CKD, truth=True, predicted=True, score=1.5
        ),
        Prediction(
            "row", 0, "d", 4, Diagnosis.CKD, truth=False, predicted=False, score=-0.5
        ),
    )
    assert report == TrainingReport(Diagnosis.CKD, 2, 2, 1)


def test_shared_subject_stops_before_saving() -> None:
    """Split sdílející pacienta vyhodí SubjectLeakError a nic se neuloží."""
    notes = (*NOTES[:3], InputNote("e", 1, "en", "text e", frozenset()))
    recorder = _Recorder()

    with pytest.raises(SubjectLeakError):
        _use_case(notes, Holdout(train=(0, 1), test=(2, 3)), recorder).run(CONFIG)

    assert "fit" not in recorder.calls
    assert "save" not in recorder.calls
    assert "predictions" not in recorder.calls


def test_class_with_too_few_patients_stops_before_splitting() -> None:
    """Diagnóza bez dost pacientů v každé třídě skončí InvalidInputError a nic se nevolá."""
    config = TrainingConfig(
        row_id="row", diagnosis=Diagnosis.AKI, seed=0, folds=2, regularization=1.0
    )
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="aki"):
        _use_case(NOTES, Holdout(train=(0, 1), test=(2, 3)), recorder).run(config)

    assert recorder.calls == {}
