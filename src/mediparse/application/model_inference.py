"""Use case inference: uložený model nad odloženými zprávami proti tréninku."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_inference import (
    ensure_predictions_present,
    ensure_same_predictions,
)

if TYPE_CHECKING:
    from mediparse.application.ports.experiment import (
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
class ModelInference:
    """Uložený model nad odloženými zprávami: soulad s tréninkem a zápis predikcí."""

    table: InputTable
    predictions: PredictionSource
    model: ModelSource
    inference: PredictionSink

    def run(self) -> tuple[Prediction, ...]:
        """Klasifikuje odložené zprávy uloženým modelem, ověří soulad a zapíše predikce.

        Returns:
            Predikce načteného modelu v pořadí predikcí tréninku.

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
            msg = (
                f"Počet odložených zpráv bez textu ve vstupní tabulce: {len(missing)}."
            )
            raise InvalidInputError(msg)
        actual = self.model.load().classify([texts[item.note_id] for item in expected])
        ensure_same_predictions(expected, actual)
        predictions = tuple(
            replace(item, predicted=result.predicted, score=result.score)
            for item, result in zip(expected, actual, strict=True)
        )
        self.inference.write(predictions)
        return predictions
