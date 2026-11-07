"""Use case inference: uložený model nad odloženými zprávami proti tréninku."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.costs import BatchThroughput, latency
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_inference import (
    ensure_predictions_present,
    ensure_same_predictions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PredictionSink,
        PredictionSource,
        TextClassifier,
    )
    from mediparse.domain.costs import Latency, LatencyPlan
    from mediparse.domain.evaluation import Prediction


class Clock(Protocol):
    """Monotonní hodiny pro měření rozdílu časů v sekundách."""

    def now(self) -> float:
        """Vrátí čas v sekundách s libovolným pevným počátkem."""


class ModelSource(Protocol):
    """Zdroj uloženého modelu."""

    def load(self) -> TextClassifier:
        """Načte uložený klasifikátor.

        Raises:
            InvalidInputError: Model chybí nebo ho nelze načíst.
        """


@dataclass(frozen=True)
class InferenceRun:
    """Predikce uloženého modelu s latencí zpráv a měřením celé dávky."""

    predictions: tuple[Prediction, ...]
    latency: Latency
    batch: BatchThroughput


@dataclass(frozen=True)
class ModelInference:
    """Uložený model nad odloženými zprávami: soulad, predikce a měření nákladů."""

    table: InputTable
    predictions: PredictionSource
    model: ModelSource
    inference: PredictionSink
    clock: Clock
    plan: LatencyPlan

    def run(self) -> InferenceRun:
        """Ověří a zapíše predikce načteného modelu a změří dávku i zahřátou inferenci.

        Returns:
            Predikce v pořadí tréninku, latence zpráv a doba klasifikace dávky.

        Raises:
            InvalidInputError: Odložená zpráva nemá text v tabulce, predikce jsou
                prázdné; ModelMismatchError, když uložený model dává jiné
                predikce než trénink.
        """
        expected = self.predictions.read()
        ensure_predictions_present(expected)
        texts = {note.note_id: note.text for note in self.table.read()}
        missing = sum(item.note_id not in texts for item in expected)
        if missing:
            # note_id nese subject_id pacienta MIMIC; hláška ani log ho nesmí nést.
            msg = f"Počet odložených zpráv bez textu ve vstupní tabulce: {missing}."
            raise InvalidInputError(msg)
        classifier = self.model.load()
        ordered_texts = [texts[item.note_id] for item in expected]
        # Dávka i jednotlivé zprávy mají měřit ustálený provoz bez prvních alokací a cache.
        for index in range(self.plan.warmup):
            classifier.classify([ordered_texts[index % len(ordered_texts)]])
        started = self.clock.now()
        actual = classifier.classify(ordered_texts)
        finished = self.clock.now()
        batch = BatchThroughput(notes=len(expected), seconds=finished - started)
        ensure_same_predictions(expected, actual)
        predictions = tuple(
            replace(item, predicted=result.predicted, score=result.score)
            for item, result in zip(expected, actual, strict=True)
        )
        self.inference.write(predictions)
        return InferenceRun(
            predictions, self._measure_latency(classifier, ordered_texts), batch
        )

    def _measure_latency(
        self, classifier: TextClassifier, texts: Sequence[str]
    ) -> Latency:
        durations = []
        # Pro provoz po jedné zprávě měříme dávku 1; celá dávka má vlastní propustnost.
        for text in texts[: self.plan.samples]:
            started = self.clock.now()
            classifier.classify([text])
            durations.append(self.clock.now() - started)
        return latency(durations)
