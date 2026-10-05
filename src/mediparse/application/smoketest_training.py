"""Use case tréninku smoketestu: klasifikátor jedné diagnózy na odložené části pacientů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.smoketest_training import (
    TrainingReport,
    ensure_disjoint_subjects,
    ensure_patients_per_class,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.smoketest_input import InputNote
    from mediparse.domain.smoketest_training import (
        BinaryMetrics,
        Holdout,
        TrainingConfig,
    )


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

    def predict(self, texts: Sequence[str]) -> tuple[bool, ...]:
        """Predikuje přítomnost diagnózy v každém textu."""


class ClassifierTrainer(Protocol):
    """Trénink klasifikátoru textu."""

    def fit(self, texts: Sequence[str], labels: Sequence[bool]) -> TextClassifier:
        """Natrénuje klasifikátor na textech a labelech."""


class ModelStore(Protocol):
    """Úložiště natrénovaného modelu."""

    def save(self, classifier: TextClassifier) -> None:
        """Uloží klasifikátor."""


class Scorer(Protocol):
    """Výpočet metrik binární klasifikace."""

    def score(self, truth: Sequence[bool], predicted: Sequence[bool]) -> BinaryMetrics:
        """Spočte metriky predikcí proti pravdě."""


class ReportSink(Protocol):
    """Výstup reportu tréninku."""

    def write(self, report: TrainingReport) -> None:
        """Zapíše report."""


@dataclass(frozen=True)
class SmoketestTraining:
    """Trénink na jedné diagnóze s held-out metrikami."""

    table: InputTable
    splitter: HoldoutSplitter
    trainer: ClassifierTrainer
    scorer: Scorer
    store: ModelStore
    report: ReportSink

    def run(self, config: TrainingConfig) -> TrainingReport:
        """Natrénuje model na trénovacích pacientech, ohodnotí ho na testovacích a uloží.

        Args:
            config: Diagnóza a parametry tréninku.

        Returns:
            Report zapsaný do výstupu.
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
        predicted = classifier.predict([notes[index].text for index in holdout.test])
        report = TrainingReport(
            diagnosis=config.diagnosis,
            train_notes=len(holdout.train),
            test_notes=len(holdout.test),
            test_positives=sum(truth),
            metrics=self.scorer.score(truth, predicted),
        )
        self.store.save(classifier)
        self.report.write(report)
        return report
