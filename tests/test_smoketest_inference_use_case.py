"""Use case inference: klasifikace odložených zpráv uloženým modelem a zápis ukázek."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.smoketest_inference import SmoketestInference
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification, Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.smoketest_inference import ModelMismatchError
from tests.model_support import prediction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import TextClassifier


NOTES = (
    InputNote("1-DS-1", 1, Language.EN, "text a", frozenset()),
    InputNote("2-DS-1", 2, Language.EN, "text b", frozenset()),
    InputNote("3-DS-1", 3, Language.EN, "text c", frozenset()),
)
HELD_OUT = (
    prediction("3-DS-1", truth=True, score=1.5, diagnosis=Diagnosis.CKD, subject_id=3),
    prediction("1-DS-1", truth=False, score=-0.5, diagnosis=Diagnosis.CKD),
)
ANSWER = (
    Classification(predicted=True, score=1.5),
    Classification(predicted=False, score=-0.5),
)


@dataclass
class _FakeTable:
    notes: Sequence[InputNote]

    def read(self) -> tuple[InputNote, ...]:
        return tuple(self.notes)


@dataclass
class _FakePredictions:
    items: Sequence[Prediction]

    def read(self) -> tuple[Prediction, ...]:
        return tuple(self.items)


@dataclass
class _FakeClassifier:
    answer: tuple[Classification, ...]
    texts: list[str] = field(default_factory=list)

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        if not texts:
            msg = "Found array with 0 sample(s)"
            raise ValueError(msg)
        self.texts.extend(texts)
        return self.answer


@dataclass
class _FakeModel:
    classifier: _FakeClassifier

    def load(self) -> TextClassifier:
        return self.classifier


@dataclass
class _FakeSink:
    written: list[tuple[Prediction, ...]] = field(default_factory=list)

    def write(self, predictions: Sequence[Prediction]) -> None:
        self.written.append(tuple(predictions))


def _use_case(
    *,
    notes: Sequence[InputNote] = NOTES,
    items: Sequence[Prediction] = HELD_OUT,
    answer: tuple[Classification, ...] = ANSWER,
) -> tuple[SmoketestInference, _FakeClassifier, _FakeSink]:
    classifier, sink = _FakeClassifier(answer), _FakeSink()
    use_case = SmoketestInference(
        _FakeTable(notes), _FakePredictions(items), _FakeModel(classifier), sink
    )
    return use_case, classifier, sink


def test_classifies_only_held_out_notes_and_writes_examples() -> None:
    """Klasifikují se jen texty odložených zpráv a zapíše se pozitivní a negativní ukázka."""
    use_case, classifier, sink = _use_case()

    result = use_case.run()

    assert classifier.texts == ["text c", "text a"]
    assert [item.note_id for item in result] == ["3-DS-1", "1-DS-1"]
    assert sink.written == [result]


def test_mismatch_raises_and_writes_nothing() -> None:
    """Nesoulad s tréninkem vyhodí chybu a ukázky se nezapíší."""
    wrong = (Classification(predicted=True, score=9.0), ANSWER[1])
    use_case, _, sink = _use_case(answer=wrong)

    with pytest.raises(ModelMismatchError, match="Nesoulad u zprávy 3-DS-1:"):
        use_case.run()

    assert sink.written == []


def test_held_out_note_without_text_raises() -> None:
    """Odložená zpráva bez textu v tabulce je neplatný vstup."""
    use_case, classifier, sink = _use_case(notes=NOTES[:2])

    with pytest.raises(InvalidInputError, match="Odložená zpráva 3-DS-1 nemá"):
        use_case.run()

    assert classifier.texts == []
    assert sink.written == []


def test_empty_predictions_raise() -> None:
    """Prázdný soubor predikcí není co ověřit."""
    use_case, _, sink = _use_case(items=(), answer=())

    with pytest.raises(InvalidInputError, match="není co ověřit"):
        use_case.run()

    assert sink.written == []
