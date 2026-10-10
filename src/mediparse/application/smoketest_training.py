"""Use case tréninku smoketestu: klasifikátor jedné diagnózy na odložené části pacientů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.evaluation import Prediction
from mediparse.domain.smoketest_training import (
    TrainingReport,
    ensure_disjoint_subjects,
    ensure_patients_per_class,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.evaluation import Classification
    from mediparse.domain.smoketest_input import InputNote
    from mediparse.domain.smoketest_training import Holdout, TrainingConfig

FOLD = 0


class InputTable(Protocol):
    """Vstupní tabulka zpráv s labely."""

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky v pořadí tabulky."""


class HoldoutSplitter(Protocol):
    """Rozdělení zpráv na trénink a test po pacientech."""

    def split(self, labels: Sequence[bool], groups: Sequence[int]) -> Holdout:
        """Rozdělí zprávy podle labelů a skupin pacientů."""


class TextClassifier(Protocol):
    """Natrénovaný binární klasifikátor textu."""

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        """Určí label a rozhodovací skóre pro každý text."""


class ClassifierTrainer(Protocol):
    """Trénink klasifikátoru textu."""

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> TextClassifier:
        """Natrénuje klasifikátor na textech a labelech."""


class ModelStore(Protocol):
    """Úložiště natrénovaného modelu."""

    def save(self, classifier: TextClassifier) -> None:
        """Uloží klasifikátor."""


class PredictionSink(Protocol):
    """Výstup predikcí testovacích zpráv."""

    def write(self, predictions: Sequence[Prediction]) -> None:
        """Zapíše predikce."""


@dataclass(frozen=True)
class SmoketestTraining:
    """Trénink na jedné diagnóze se zápisem predikcí odložených zpráv."""

    table: InputTable
    splitter: HoldoutSplitter
    trainer: ClassifierTrainer
    store: ModelStore
    predictions: PredictionSink

    def run(self, config: TrainingConfig) -> TrainingReport:
        """Natrénuje model na trénovacích pacientech, predikuje testovací a uloží.

        Args:
            config: Diagnóza a parametry tréninku.

        Returns:
            Velikosti částí odložení.
        """
        notes = self.table.read()
        labels = [config.diagnosis in note.labels for note in notes]
        groups = [note.subject_id for note in notes]
        ensure_patients_per_class(labels, groups, config.diagnosis, config.folds)
        holdout = self.splitter.split(labels, groups)
        ensure_disjoint_subjects(notes, holdout)
        classifier = self.trainer.fit(
            [notes[index].text for index in holdout.train],
            [labels[index] for index in holdout.train],
        )
        truth = [labels[index] for index in holdout.test]
        classifications = classifier.classify([
            notes[index].text for index in holdout.test
        ])
        predictions = [
            Prediction(
                config.row_id,
                FOLD,
                notes[index].note_id,
                notes[index].subject_id,
                config.diagnosis,
                truth=label,
                predicted=classification.predicted,
                score=classification.score,
            )
            for index, label, classification in zip(
                holdout.test, truth, classifications, strict=True
            )
        ]
        self.store.save(classifier)
        self.predictions.write(predictions)
        return TrainingReport(
            diagnosis=config.diagnosis,
            train_notes=len(holdout.train),
            test_notes=len(holdout.test),
            test_positives=sum(truth),
        )
