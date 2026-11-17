"""Use case tréninku modelu: klasifikátor jedné diagnózy na odložené části pacientů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.ports.experiment import TextClassifier
from mediparse.domain.evaluation import Prediction
from mediparse.domain.split import ensure_patients_per_class
from mediparse.domain.training import TrainingReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PredictionSink,
        SplitSource,
    )
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.training import TrainingConfig


def _labels(notes: Sequence[InputNote], diagnosis: Diagnosis) -> list[bool]:
    return [note.has_label(diagnosis) for note in notes]


class TrainedClassifier(TextClassifier, Protocol):
    """Výsledek tréninku s prahem a konvergencí pro provenance běhu."""

    @property
    def threshold(self) -> float:
        """Konečný práh na stejné škále jako rozhodovací skóre."""

    @property
    def converged(self) -> bool:
        """Zda konvergovaly všechny fity včetně vnitřní CV."""


class ClassifierTrainer[C: TrainedClassifier](Protocol):
    """Trénink klasifikátoru textu."""

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[bool],
        subjects: Sequence[SubjectId],
    ) -> C:
        """Natrénuje klasifikátor na textech a labelech."""


class ModelStore[C: TextClassifier](Protocol):
    """Úložiště modelu, které umí uložit klasifikátor typu ``C``."""

    def save(self, classifier: C) -> None:
        """Uloží klasifikátor."""


@dataclass(frozen=True)
class ModelTraining[C: TrainedClassifier]:
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
            Velikosti částí, původ splitu, zvolený práh a konvergence.
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
        train_labels = _labels(holdout.train, config.diagnosis)
        train_subjects = [note.subject_id for note in holdout.train]
        ensure_patients_per_class(
            train_labels,
            train_subjects,
            config.diagnosis,
            config.threshold_folds,
        )
        classifier = self.trainer.fit(
            [note.text for note in holdout.train],
            train_labels,
            train_subjects,
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
            threshold=classifier.threshold,
            converged=classifier.converged,
        )
