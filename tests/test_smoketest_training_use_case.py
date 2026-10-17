"""Use case tréninku: labely zvolené diagnózy, trénink jen na train, predikce jen testu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.smoketest_training import SmoketestTraining
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification, Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.smoketest_training import TrainingReport
from mediparse.domain.split import Split, strata
from tests.model_support import parsed_training_config

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIG = parsed_training_config(diagnosis="ckd", row_id="row")
SPLIT = Split((frozenset({1, 2}), frozenset({3, 4})))
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
    split_result: Split
    recorder: _Recorder

    def split(self, subjects: Sequence[int], strata: Sequence[int]) -> Split:
        self.recorder.calls["split"] = (tuple(subjects), tuple(strata))
        return self.split_result


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
    notes: Sequence[InputNote], split: Split, recorder: _Recorder
) -> SmoketestTraining[_FakeClassifier]:
    return SmoketestTraining(
        table=_FakeTable(notes),
        splitter=_FakeSplitter(split, recorder),
        trainer=_FakeTrainer(recorder),
        store=_FakeStore(recorder),
        predictions=_FakeSink(recorder),
    )


def test_trains_on_train_and_predicts_test() -> None:
    """Labely jsou zvolené diagnózy; trénink vidí jen train, predikce jen test."""
    recorder = _Recorder()

    report = _use_case(NOTES, SPLIT, recorder).run(CONFIG)

    assert recorder.calls["split"] == ((1, 2, 3, 4), strata(NOTES, CONFIG.folds))
    assert recorder.calls["fit"] == (("text c", "text d"), (True, False))
    assert recorder.calls["classify"] == ("text a", "text b")
    assert isinstance(recorder.calls["save"], _FakeClassifier)
    assert recorder.calls["predictions"] == (
        Prediction(
            "row", 0, "a", 1, Diagnosis.CKD, truth=True, predicted=True, score=1.5
        ),
        Prediction(
            "row", 0, "b", 2, Diagnosis.CKD, truth=False, predicted=False, score=-0.5
        ),
    )
    assert report == TrainingReport(Diagnosis.CKD, 2, 2, 1)


def test_test_fold_comes_from_config() -> None:
    """Fold z configu určí testovací pacienty a nese ho každá predikce."""
    recorder = _Recorder()
    config = parsed_training_config(diagnosis="ckd", row_id="row", fold=1)

    _use_case(NOTES, SPLIT, recorder).run(config)

    assert recorder.calls["classify"] == ("text c", "text d")
    assert recorder.calls["fit"] == (("text a", "text b"), (True, False))
    predictions = recorder.calls["predictions"]
    assert isinstance(predictions, tuple)
    assert {(item.fold, item.note_id) for item in predictions} == {(1, "c"), (1, "d")}


@pytest.mark.parametrize("fold", [2, 7])
def test_fold_out_of_range_stops_before_saving(fold: int) -> None:
    """Fold mimo split je InvalidInputError a nic se neuloží."""
    recorder = _Recorder()
    config = parsed_training_config(diagnosis="ckd", row_id="row", fold=fold)

    with pytest.raises(InvalidInputError, match=rf"Fold {fold} neexistuje"):
        _use_case(NOTES, SPLIT, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}


def test_subject_outside_split_stops_before_saving() -> None:
    """Pacient mimo split je InvalidInputError a nic se neuloží."""
    notes = (*NOTES, InputNote("e", 5, Language.EN, "text e", frozenset()))
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="Zprávy pacientů mimo split"):
        _use_case(notes, SPLIT, recorder).run(CONFIG)

    assert recorder.calls.keys() <= {"split"}


def test_class_with_too_few_patients_stops_before_splitting() -> None:
    """Diagnóza bez dost pacientů v každé třídě skončí InvalidInputError a nic se nevolá."""
    config = parsed_training_config(diagnosis="pneumonia", row_id="row")
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="pneumonia"):
        _use_case(NOTES, SPLIT, recorder).run(config)

    assert recorder.calls == {}
