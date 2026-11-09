"""Use case tréninku modelu: klasifikátor jedné diagnózy na odložené části pacientů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.evaluation import Prediction
from mediparse.domain.split import ensure_patients_per_class
from mediparse.domain.training import TrainingReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PredictionSink,
        SplitSource,
        TextClassifier,
    )
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.training import TrainingConfig


def _labels(notes: Sequence[InputNote], diagnosis: Diagnosis) -> list[bool]:
    return [note.has_label(diagnosis) for note in notes]


class ClassifierTrainer[C: TextClassifier](Protocol):
    """Trénink klasifikátoru textu."""

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> C:
        """Natrénuje klasifikátor na textech a labelech."""


class ModelStore[C: TextClassifier](Protocol):
    """Úložiště modelu, které umí uložit klasifikátor typu ``C``."""

    def save(self, classifier: C) -> None:
        """Uloží klasifikátor."""


@dataclass(frozen=True)
class ModelTraining[C: TextClassifier]:
    """Trénink na jedné diagnóze se zápisem predikcí odložených zpráv.

    Trenér a úložiště sdílejí typ klasifikátoru, takže typová kontrola odmítne
    úložiště, které by model z trenéra neumělo uložit.
    """

    table: InputTable
    split: SplitSource
    trainer: ClassifierTrainer[C]
    store: ModelStore[C]
    predictions: PredictionSink

    def run(self, config: TrainingConfig) -> TrainingReport:
        """Natrénuje model na trénovacích pacientech, predikuje testovací a uloží.

        Args:
            config: Diagnóza a parametry tréninku.

        Returns:
            Velikosti částí odložení, config a otisk splitu.
        """
        notes = self.table.read()
        shared = self.split.read()
        holdout = shared.split.holdout(notes, config.fold)
        evaluation_notes = holdout.train + holdout.test
        ensure_patients_per_class(
            _labels(evaluation_notes, config.diagnosis),
            [note.subject_id for note in evaluation_notes],
            config.diagnosis,
            shared.config.folds,
        )
        classifier = self.trainer.fit(
            [note.text for note in holdout.train],
            _labels(holdout.train, config.diagnosis),
        )
        classifications = classifier.classify([note.text for note in holdout.test])
        predictions = [
            Prediction(
                config.row_id,
                holdout.fold,
                note.note_id,
                note.subject_id,
                config.diagnosis,
                predicted=classification.predicted,
                score=classification.score,
            )
            for note, classification in zip(holdout.test, classifications, strict=True)
        ]
        self.store.save(classifier)
        self.predictions.write(predictions)
        return TrainingReport(
            diagnosis=config.diagnosis,
            train_notes=len(holdout.train),
            test_notes=len(holdout.test),
            test_positives=sum(_labels(holdout.test, config.diagnosis)),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )
