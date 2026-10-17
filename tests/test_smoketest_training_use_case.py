"""Use case tréninku: labely zvolené diagnózy, trénink jen na train, predikce jen testu."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.smoketest_training import SmoketestTraining
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification, Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.smoketest_training import TrainingConfig, TrainingReport
from mediparse.domain.split import Holdout, SubjectLeakError
from tests.model_support import training_config

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIG = TrainingConfig.model_validate_json(
    json.dumps(training_config(diagnosis="ckd"))
)
NOTES = (
    InputNote("a", 1, Language.EN, "text a", frozenset({Diagnosis.CKD})),
    InputNote("b", 2, Language.EN, "text b", frozenset({Diagnosis.DIABETES})),
    InputNote("c", 3, Language.EN, "text c", frozenset({Diagnosis.CKD})),
    InputNote("d", 4, Language.EN, "text d", frozenset()),
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

    def save(self, classifier: _FakeClassifier) -> None:
        self.recorder.calls["save"] = classifier


@dataclass
class _FakeSink:
    recorder: _Recorder

    def write(self, predictions: Sequence[Prediction]) -> None:
        self.recorder.calls["predictions"] = tuple(predictions)


def _use_case(
    notes: Sequence[InputNote], holdout: Holdout, recorder: _Recorder
) -> SmoketestTraining[_FakeClassifier]:
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
    notes = (*NOTES[:3], InputNote("e", 1, Language.EN, "text e", frozenset()))
    recorder = _Recorder()

    with pytest.raises(SubjectLeakError):
        _use_case(notes, Holdout(train=(0, 1), test=(2, 3)), recorder).run(CONFIG)

    assert "fit" not in recorder.calls
    assert "save" not in recorder.calls
    assert "predictions" not in recorder.calls


def test_class_with_too_few_patients_stops_before_splitting() -> None:
    """Diagnóza bez dost pacientů v každé třídě skončí InvalidInputError a nic se nevolá."""
    config = TrainingConfig.model_validate_json(
        json.dumps(training_config(diagnosis="pneumonia"))
    )
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="pneumonia"):
        _use_case(NOTES, Holdout(train=(0, 1), test=(2, 3)), recorder).run(config)

    assert recorder.calls == {}
