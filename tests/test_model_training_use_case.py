"""Use case tréninku: labely zvolené diagnózy, trénink jen na train, predikce jen testu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.model_training import ModelTraining
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

CONFIG = parsed_training_config(diagnosis="ckd", row_id="row", threshold_folds=2)
SPLIT = Split(
    (frozenset({1, 2, 5, 6}), frozenset({3, 4, 7, 8})), development=frozenset()
)
SPLIT_CONFIG = split_config(folds=2, seed=4)
SHARED = SharedSplit(SPLIT_CONFIG, SPLIT)
NOTES = (
    InputNote("1-DS-1", 1, Language.EN, "text a", frozenset({Diagnosis.CKD})),
    InputNote("2-DS-1", 2, Language.EN, "text b", frozenset({Diagnosis.DIABETES})),
    InputNote("3-DS-1", 3, Language.EN, "text c", frozenset({Diagnosis.CKD})),
    InputNote("4-DS-1", 4, Language.EN, "text d", frozenset()),
    InputNote("5-DS-1", 5, Language.EN, "text e", frozenset({Diagnosis.CKD})),
    InputNote("6-DS-1", 6, Language.EN, "text f", frozenset()),
    InputNote("7-DS-1", 7, Language.EN, "text g", frozenset({Diagnosis.CKD})),
    InputNote("8-DS-1", 8, Language.EN, "text h", frozenset()),
)


ANSWER = (
    Classification(predicted=False, score=-1.5),
    Classification(predicted=True, score=0.5),
) * 2


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
    threshold: float = 0.25
    converged: bool = False

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        self.recorder.calls["classify"] = tuple(texts)
        return self.answer


@dataclass
class _FakeTrainer:
    recorder: _Recorder
    answer: tuple[Classification, ...] = ANSWER

    def fit(
        self, texts: Sequence[str], labels: Sequence[bool], subjects: Sequence[int]
    ) -> _FakeClassifier:
        self.recorder.calls["fit"] = (tuple(texts), tuple(labels), tuple(subjects))
        return _FakeClassifier(self.answer, self.recorder)


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
    notes: Sequence[InputNote],
    shared: SharedSplit,
    recorder: _Recorder,
    *,
    answer: tuple[Classification, ...] = ANSWER,
) -> ModelTraining[_FakeClassifier]:
    return ModelTraining(
        table=_FakeTable(notes),
        split=_FakeSplitSource(shared, recorder),
        trainer=_FakeTrainer(recorder, answer),
        store=_FakeStore(recorder),
        predictions=_FakeSink(recorder),
    )


def test_trains_on_train_and_predicts_test() -> None:
    """Labely jsou zvolené diagnózy; trénink vidí jen train, predikce jen test."""
    recorder = _Recorder()

    report = _use_case(NOTES, SHARED, recorder).run(CONFIG)

    assert recorder.calls["fit"] == (
        ("text c", "text d", "text g", "text h"),
        (True, False, True, False),
        (3, 4, 7, 8),
    )
    assert recorder.calls["classify"] == ("text a", "text b", "text e", "text f")
    assert isinstance(recorder.calls["save"], _FakeClassifier)
    assert recorder.calls["predictions"] == tuple(
        Prediction(
            "row", 0, note_id, subject, Diagnosis.CKD, predicted=predicted, score=score
        )
        for note_id, subject, predicted, score in (
            ("1-DS-1", 1, False, -1.5),
            ("2-DS-1", 2, True, 0.5),
            ("5-DS-1", 5, False, -1.5),
            ("6-DS-1", 6, True, 0.5),
        )
    )
    assert report == TrainingReport(
        Diagnosis.CKD,
        train_notes=4,
        test_notes=4,
        test_positives=2,
        split_config=SPLIT_CONFIG,
        split_sha256=SPLIT.sha256(),
        threshold=0.25,
        converged=False,
    )


@pytest.mark.parametrize("fold", [0, 1])
def test_training_and_predictions_exclude_development(fold: int) -> None:
    """Jedinečné syntetické vývojové texty ani jejich zprávy nedostane trenér či sink."""
    development = (
        InputNote("9-DS-1", 9, Language.EN, "syntetický vývoj první", frozenset()),
        InputNote("9-DS-2", 9, Language.EN, "syntetický vývoj druhý", frozenset()),
    )
    shared = SharedSplit(SPLIT_CONFIG, Split(SPLIT.folds, development=frozenset({9})))
    recorder = _Recorder()
    config = parsed_training_config(
        diagnosis="ckd", row_id="row", fold=fold, threshold_folds=2
    )

    _use_case((*NOTES, *development), shared, recorder).run(config)

    fit = recorder.calls["fit"]
    assert isinstance(fit, tuple)
    texts, _, _ = fit
    assert all(note.text not in texts for note in development)
    predictions = recorder.calls["predictions"]
    assert isinstance(predictions, tuple)
    assert {item.note_id for item in predictions}.isdisjoint(
        note.note_id for note in development
    )


def test_development_positives_do_not_satisfy_class_check() -> None:
    """Dva pozitivní pacienti jen ve vývoji nestačí; foldy mají nula pozitivních."""
    development = (
        InputNote(
            "9-DS-1",
            9,
            Language.EN,
            "syntetický vývoj",
            frozenset({Diagnosis.PNEUMONIA}),
        ),
        InputNote(
            "10-DS-1",
            10,
            Language.EN,
            "syntetický vývoj",
            frozenset({Diagnosis.PNEUMONIA}),
        ),
    )
    shared = SharedSplit(
        SPLIT_CONFIG, Split(SPLIT.folds, development=frozenset({9, 10}))
    )
    recorder = _Recorder()
    config = parsed_training_config(
        diagnosis="pneumonia", row_id="row", threshold_folds=2
    )

    with pytest.raises(
        InvalidInputError, match=r"pneumonia.*0 pozitivních a 8 negativních"
    ):
        _use_case((*NOTES, *development), shared, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}


def test_too_few_classifications_stop_before_saving() -> None:
    """Chybějící klasifikace testovací zprávy vyvolá chybu před uložením výstupů."""
    recorder = _Recorder()
    use_case = _use_case(NOTES, SHARED, recorder, answer=ANSWER[:1])

    with pytest.raises(
        ValueError, match=r"zip\(\) argument 2 is shorter than argument 1"
    ):
        use_case.run(CONFIG)

    assert "save" not in recorder.calls
    assert "predictions" not in recorder.calls


def test_test_fold_comes_from_config() -> None:
    """Fold z configu určí testovací pacienty a nese ho každá predikce."""
    recorder = _Recorder()
    config = parsed_training_config(
        diagnosis="ckd", row_id="row", fold=1, threshold_folds=2
    )

    _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls["classify"] == ("text c", "text d", "text g", "text h")
    assert recorder.calls["fit"] == (
        ("text a", "text b", "text e", "text f"),
        (True, False, True, False),
        (1, 2, 5, 6),
    )
    predictions = recorder.calls["predictions"]
    assert isinstance(predictions, tuple)
    assert {(item.fold, item.note_id) for item in predictions} == {
        (1, "3-DS-1"),
        (1, "4-DS-1"),
        (1, "7-DS-1"),
        (1, "8-DS-1"),
    }


@pytest.mark.parametrize("fold", [2, 7])
def test_fold_out_of_range_stops_before_saving(fold: int) -> None:
    """Fold mimo split je InvalidInputError a nic se neuloží."""
    recorder = _Recorder()
    config = parsed_training_config(
        diagnosis="ckd", row_id="row", fold=fold, threshold_folds=2
    )

    with pytest.raises(InvalidInputError, match=rf"Fold {fold} neexistuje"):
        _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}


def test_subject_outside_split_stops_before_saving() -> None:
    """Pacient mimo split je InvalidInputError a nic se neuloží."""
    notes = (*NOTES, InputNote("9-DS-1", 9, Language.EN, "text i", frozenset()))
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match=r"^Pacienti mimo split: 1\.$"):
        _use_case(notes, SHARED, recorder).run(CONFIG)

    assert recorder.calls.keys() <= {"split"}


def test_class_with_too_few_patients_stops_before_training() -> None:
    """Diagnóza bez dost pacientů v každé třídě skončí InvalidInputError a trénink se nespustí."""
    config = parsed_training_config(
        diagnosis="pneumonia", row_id="row", threshold_folds=2
    )
    recorder = _Recorder()

    with pytest.raises(InvalidInputError, match="pneumonia"):
        _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}


def test_class_check_uses_folds_of_shared_split() -> None:
    """Počet pacientů ve třídě se měří proti ``folds`` z configu sdíleného splitu."""
    recorder = _Recorder()
    shared = SharedSplit(
        split_config(folds=3),
        Split(
            (frozenset({1}), frozenset({2}), frozenset({3, 4})), development=frozenset()
        ),
    )

    with pytest.raises(InvalidInputError, match=r"ckd.*aspoň 3\."):
        _use_case(NOTES[:4], shared, recorder).run(CONFIG)

    assert recorder.calls.keys() <= {"split"}


def test_threshold_class_check_uses_only_outer_training_patients() -> None:
    """Dvě trénovací pozitivity nestačí pro tři vnitřní foldy, i když je celý split má."""
    recorder = _Recorder()
    config = parsed_training_config(diagnosis="ckd", threshold_folds=3)

    with pytest.raises(
        InvalidInputError, match=r"ckd.*2 pozitivních a 2 negativních.*aspoň 3"
    ):
        _use_case(NOTES, SHARED, recorder).run(config)

    assert recorder.calls.keys() <= {"split"}
