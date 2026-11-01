"""Use case inference: klasifikace odložených zpráv uloženým modelem a zápis predikcí."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import pytest

from mediparse.application.model_inference import ModelInference
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification, Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.model_inference import ModelMismatchError
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
    prediction("3-DS-1", truth=True, score=1.5, diagnosis=Diagnosis.CKD),
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
) -> tuple[ModelInference, _FakeClassifier, _FakeSink]:
    classifier, sink = _FakeClassifier(answer), _FakeSink()
    use_case = ModelInference(
        _FakeTable(notes), _FakePredictions(items), _FakeModel(classifier), sink
    )
    return use_case, classifier, sink


def test_classifies_only_held_out_notes() -> None:
    """Text zprávy mimo odloženou část se do klasifikátoru nedostane."""
    use_case, classifier, _ = _use_case()

    use_case.run()

    assert classifier.texts == ["text c", "text a"]


def test_mismatch_raises_and_writes_nothing() -> None:
    """Nesoulad s tréninkem vyhodí chybu a predikce se nezapíší."""
    wrong = (Classification(predicted=True, score=9.0), ANSWER[1])
    use_case, _, sink = _use_case(answer=wrong)

    with pytest.raises(ModelMismatchError, match=r"Nesoulad u 1\. predikce z 2:"):
        use_case.run()

    assert sink.written == []


def test_held_out_note_without_text_raises() -> None:
    """Odložená zpráva bez textu v tabulce je neplatný vstup."""
    use_case, classifier, sink = _use_case(notes=NOTES[:2])

    with pytest.raises(
        InvalidInputError,
        match="Počet odložených zpráv bez textu ve vstupní tabulce: 1",
    ):
        use_case.run()

    assert classifier.texts == []
    assert sink.written == []


def test_empty_predictions_raise() -> None:
    """Prázdný soubor predikcí není co ověřit."""
    use_case, _, sink = _use_case(items=(), answer=())

    with pytest.raises(InvalidInputError, match="není co ověřit"):
        use_case.run()

    assert sink.written == []


def test_missing_texts_report_only_count() -> None:
    """Dva chybějící texty ze tří syntetických zpráv neprozradí identifikátory."""
    items = tuple(
        prediction(f"9000000{index}-DS-1", truth=True, score=0.5) for index in (1, 2, 3)
    )
    notes = (InputNote("90000001-DS-1", 90000001, Language.EN, "text", frozenset()),)
    use_case, classifier, sink = _use_case(notes=notes, items=items)

    with pytest.raises(InvalidInputError) as raised:
        use_case.run()

    message = str(raised.value)
    assert "2" in message
    assert "9000000" not in message
    assert "DS-" not in message
    assert classifier.texts == []
    assert sink.written == []


def test_writes_all_reloaded_predictions_in_training_order() -> None:
    """Inference zachová metadata a pořadí, skóre však pocházejí z načteného modelu."""
    items = (*HELD_OUT, prediction("2-DS-1", truth=True, score=2.0))
    answer = (*ANSWER, Classification(predicted=True, score=2.0 + 1e-12))
    use_case, classifier, sink = _use_case(items=items, answer=answer)

    result = use_case.run()

    assert classifier.texts == ["text c", "text a", "text b"]
    assert result == (*HELD_OUT, replace(items[2], score=2.0 + 1e-12))
    assert sink.written == [result]
