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
from mediparse.domain.split import SharedSplit, Split
from mediparse.domain.training import TrainingReport
from tests.model_support import parsed_training_config, split_config

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIG = parsed_training_config(diagnosis="ckd", row_id="row")
SPLIT = Split((frozenset({1, 2}), frozenset({3, 4})))
SPLIT_CONFIG = split_config(folds=2, seed=4)
SHARED = SharedSplit(SPLIT_CONFIG, SPLIT)
NOTES = (
    InputNote("1-DS-1", 1, Language.EN, "text a", frozenset({Diagnosis.CKD})),
    InputNote("2-DS-1", 2, Language.EN, "text b", frozenset({Diagnosis.DIABETES})),
    InputNote("3-DS-1", 3, Language.EN, "text c", frozenset({Diagnosis.CKD})),
    InputNote("4-DS-1", 4, Language.EN, "text d", frozenset()),
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
class _FakeSplitSource:
    shared: SharedSplit
    recorder: _Recorder

    def read(self) -> SharedSplit:
        self.recorder.calls["split"] = True
        return self.shared


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
    notes: Sequence[InputNote], shared: SharedSplit, recorder: _Recorder
) -> SmoketestTraining[_FakeClassifier]:
    return SmoketestTraining(
        table=_FakeTable(notes),
        split=_FakeSplitSource(shared, recorder),
        trainer=_FakeTrainer(recorder),
        store=_FakeStore(recorder),
        predictions=_FakeSink(recorder),
    )


def test_trains_on_train_and_predicts_test() -> None:
    """Labely jsou zvolené diagnózy; trénink vidí jen train, predikce jen test."""
    recorder = _Recorder()

    report = _use_case(NOTES, SHARED, recorder).run(CONFIG)

    assert recorder.calls["fit"] == (("text c", "text d"), (True, False))
    assert recorder.calls["classify"] == ("text a", "text b")
    assert isinstance(recorder.calls["save"], _FakeClassifier)
    assert recorder.calls["predictions"] == (
        Prediction(
            "row", 0, "1-DS-1", 1, Diagnosis.CKD, truth=True, predicted=True, score=1.5
        ),
        Prediction(
            "row",
            0,
            "2-DS-1",
            2,
            Diagnosis.CKD,
            truth=False,
            predicted=False,
            score=-0.5,
        ),
    )
    assert report == TrainingReport(
        Diagnosis.CKD,
        train_notes=2,
        test_notes=2,
        test_positives=1,
        split_config=SPLIT_CONFIG,
        split_sha256=SPLIT.sha256(),
    )


def test_test_fold_comes_from_config() -> None:
    """Fold z configu určí testovací pacienty a nese ho každá predikce."""
    recorder = _Recorder()
    config = parsed_training_config(diagnosis="ckd", row_id="row", fold=1)

    _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls["classify"] == ("text c", "text d")
    assert recorder.calls["fit"] == (("text a", "text b"), (True, False))
    predictions = recorder.calls["predictions"]
    assert isinstance(predictions, tuple)
    assert {(item.fold, item.note_id) for item in predictions} == {
        (1, "3-DS-1"),
        (1, "4-DS-1"),
    }


@pytest.mark.parametrize("fold", [2, 7])
def test_fold_out_of_range_stops_before_saving(fold: int) -> None:
    """Fold mimo split je InvalidInputError a nic se neuloží."""
    recorder = _Recorder()
    config = parsed_training_config(diagnosis="ckd", row_id="row", fold=fold)

    with pytest.raises(InvalidInputError, match=rf"Fold {fold} neexistuje"):
        _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}


def test_subject_outside_split_stops_before_saving() -> None:
    """Pacient mimo split je InvalidInputError a nic se neuloží."""
    notes = (*NOTES, InputNote("5-DS-1", 5, Language.EN, "text e", frozenset()))
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="Zprávy pacientů mimo split"):
        _use_case(notes, SHARED, recorder).run(CONFIG)

    assert recorder.calls.keys() <= {"split"}


def test_class_with_too_few_patients_stops_before_training() -> None:
    """Diagnóza bez dost pacientů v každé třídě skončí InvalidInputError a trénink se nespustí."""
    config = parsed_training_config(diagnosis="pneumonia", row_id="row")
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="pneumonia"):
        _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}


def test_class_check_uses_folds_of_shared_split() -> None:
    """Počet pacientů ve třídě se měří proti ``folds`` z configu sdíleného splitu."""
    recorder = _Recorder()
    shared = SharedSplit(
        split_config(folds=3),
        Split((frozenset({1}), frozenset({2}), frozenset({3, 4}))),
    )

    with pytest.raises(InvalidInputError, match=r"ckd.*aspoň 3\."):
        _use_case(NOTES, shared, recorder).run(CONFIG)

    assert recorder.calls.keys() <= {"split"}
