"""Use case tréninku smoketestu: klasifikátor jedné diagnózy na odložené části pacientů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.evaluation import Prediction
from mediparse.domain.smoketest_training import TrainingReport
from mediparse.domain.split import ensure_patients_per_class, strata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PredictionSink,
        TextClassifier,
    )
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.smoketest_training import TrainingConfig
    from mediparse.domain.split import Split


def _labels(notes: Sequence[InputNote], diagnosis: Diagnosis) -> list[bool]:
    return [diagnosis in note.labels for note in notes]


class PatientSplitter(Protocol):
    """Rozdělení pacientů do foldů."""

    def split(self, subjects: Sequence[SubjectId], strata: Sequence[int]) -> Split:
        """Rozdělí pacienty zpráv do foldů podle třídy stratifikace každé zprávy."""


class ClassifierTrainer[C: TextClassifier](Protocol):
    """Trénink klasifikátoru textu."""

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> C:
        """Natrénuje klasifikátor na textech a labelech."""


class ModelStore[C: TextClassifier](Protocol):
    """Úložiště modelu, které umí uložit klasifikátor typu ``C``."""

    def save(self, classifier: C) -> None:
        """Uloží klasifikátor."""


@dataclass(frozen=True)
class SmoketestTraining[C: TextClassifier]:
    """Trénink na jedné diagnóze se zápisem predikcí odložených zpráv.

    Trenér a úložiště sdílejí typ klasifikátoru, takže typová kontrola odmítne
    úložiště, které by model z trenéra neumělo uložit.
    """

    table: InputTable
    splitter: PatientSplitter
    trainer: ClassifierTrainer[C]
    store: ModelStore[C]
    predictions: PredictionSink

    def run(self, config: TrainingConfig) -> TrainingReport:
        """Natrénuje model na trénovacích pacientech, predikuje testovací a uloží.

        Args:
            config: Diagnóza a parametry tréninku.

        Returns:
            Velikosti částí odložení.
        """
        notes = self.table.read()
        labels = _labels(notes, config.diagnosis)
        subjects = [note.subject_id for note in notes]
        ensure_patients_per_class(labels, subjects, config.diagnosis, config.folds)
        split = self.splitter.split(subjects, strata(notes, config.folds))
        holdout = split.holdout(notes, config.fold)
        classifier = self.trainer.fit(
            [note.text for note in holdout.train],
            _labels(holdout.train, config.diagnosis),
        )
        truth = _labels(holdout.test, config.diagnosis)
        classifications = classifier.classify([note.text for note in holdout.test])
        predictions = [
            Prediction(
                config.row_id,
                holdout.fold,
                note.note_id,
                note.subject_id,
                config.diagnosis,
                truth=label,
                predicted=classification.predicted,
                score=classification.score,
            )
            for note, label, classification in zip(
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
