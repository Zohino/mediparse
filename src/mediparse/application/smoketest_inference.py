"""Use case inference smoketestu: uložený model nad odloženými zprávami proti tréninku."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.smoketest_inference import (
    demo_examples,
    ensure_predictions_present,
    ensure_same_predictions,
)

if TYPE_CHECKING:
    from mediparse.application.ports import (
        InputTable,
        PredictionSink,
        PredictionSource,
        TextClassifier,
    )
    from mediparse.domain.evaluation import Prediction


class ModelSource(Protocol):
    """Zdroj uloženého modelu."""

    def load(self) -> TextClassifier:
        """Načte uložený klasifikátor.

        Raises:
            InvalidInputError: Model chybí nebo ho nelze načíst.
        """


@dataclass(frozen=True)
class SmoketestInference:
    """Uložený model nad odloženými zprávami: shoda s tréninkem a zápis ukázek."""

    table: InputTable
    predictions: PredictionSource
    model: ModelSource
    demo: PredictionSink

    def run(self) -> tuple[Prediction, ...]:
        """Klasifikuje odložené zprávy uloženým modelem, ověří shodu a zapíše ukázky.

        Returns:
            Zapsané ukázky: první pozitivní a první negativní odložená zpráva.

        Raises:
            InvalidInputError: Odložená zpráva nemá text v tabulce, predikce jsou
                prázdné; ModelMismatchError, když uložený model dává jiné
                predikce než trénink.
        """
        expected = self.predictions.read()
        ensure_predictions_present(expected)
        texts = {note.note_id: note.text for note in self.table.read()}
        missing = [item.note_id for item in expected if item.note_id not in texts]
        if missing:
            msg = f"Odložená zpráva {missing[0]} nemá text ve vstupní tabulce."
            raise InvalidInputError(msg)
        actual = self.model.load().classify([texts[item.note_id] for item in expected])
        ensure_same_predictions(expected, actual)
        examples = demo_examples(expected)
        self.demo.write(examples)
        return examples
