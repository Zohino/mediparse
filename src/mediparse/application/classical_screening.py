"""Screening klasické mřížky s vnější CV jen mezi vývojovými pacienty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.patient_cv import patient_holdouts
from mediparse.domain.classical_screening import (
    ScreeningReport,
    ScreeningResult,
    ScreeningUnit,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.split import ensure_patients_per_class

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        PatientNoteTable,
        PatientPartitioner,
        Scorer,
        SplitSource,
    )
    from mediparse.domain.classical_screening import (
        ClassicalScreeningConfig,
        ClassifierFamily,
        VectorizerSpec,
    )
    from mediparse.domain.evaluation import Classification
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import Holdout


class Featurizer[F](Protocol):
    """Příznaky fitované jen na tréninku, sdílené rodinami a diagnózami foldu."""

    def fit_transform(
        self, spec: VectorizerSpec, train: Sequence[str], test: Sequence[str]
    ) -> tuple[F, F]:
        """Fitne vektorizér na tréninku a vrátí příznaky tréninku i testu."""


class TunedClassifier[F](Protocol):
    """Binární klasifikátor s prahem z vnitřní CV tréninku vnějšího foldu."""

    @property
    def threshold(self) -> float:
        """Rozhodovací práh na škále skóre klasifikátoru."""

    @property
    def converged(self) -> bool:
        """Zda konvergovaly všechny fity této buňky včetně volby prahu."""

    def classify(self, features: F) -> tuple[Classification, ...]:
        """Vrátí predikce při zvoleném prahu a skóre na téže škále."""


class ThresholdTuner[F](Protocol):
    """Volba prahu a refit rodiny na celé tréninkové části vnějšího foldu."""

    # Pacienti i labely zůstávají samostatné vstupy CV, adaptér proto nemusí číst zprávy.
    def fit(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        family: ClassifierFamily,
        features: F,
        labels: Sequence[bool],
        subjects: Sequence[SubjectId],
        folds: int,
        seed: Seed,
    ) -> TunedClassifier[F]:
        """Zvolí práh vnitřní CV po pacientech bez přístupu k testu vnějšího foldu."""


class ScreeningProgress(Protocol):
    """Rozpracovaný běh screeningu: hotové jednotky, jejich trvalý zápis a složení výstupů."""

    def completed(self) -> frozenset[ScreeningUnit]:
        """Ověří, že rozpracovaný běh patří tomuto screeningu, a vrátí jeho hotové jednotky."""

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Trvale uloží výsledky a predikce jedné jednotky, než vrátí řízení."""

    def finish(self) -> None:
        """Složí výstupy ze všech jednotek v pořadí zápisu a rozpracovaný běh odstraní."""


@dataclass(frozen=True)
class ClassicalScreening[F]:
    """Provede mřížku klasického stupně na vývojové části původního splitu."""

    table: PatientNoteTable
    split: SplitSource
    partitioner: PatientPartitioner
    featurizer: Featurizer[F]
    tuner: ThresholdTuner[F]
    scorer: Scorer
    progress: ScreeningProgress

    def run(self, config: ClassicalScreeningConfig) -> ScreeningReport:
        """Ověří foldy, přeskočí hotové jednotky rozpracovaného běhu, ostatní zapíše po jednotkách a složí výstupy.

        Args:
            config: Mřížka v pořadí výpočtu, parametry CV a seed screeningu.

        Returns:
            Velikost vývojové části a provenance původního splitu.

        Raises:
            InvalidInputError: Pacient tabulky je mimo split, vývojová část nemá dost
                pacientů v každé třídě, některý test foldu nesplňuje ``MIN_CELL``
                nebo trénink foldu nemá dost pacientů pro vnitřní CV prahu;
                rozpracovaný běh patří jinému běhu nebo je poškozený.
        """
        shared = self.split.read()
        shared.split.ensure_covers(self.table.subjects())
        notes = self.table.read_patients(shared.split.development)
        holdouts = _development_holdouts(notes, config, self.partitioner)
        done = self.progress.completed()
        for holdout in holdouts:
            for spec in config.vectorizers:
                families = [
                    family
                    for family in config.families
                    if ScreeningUnit(holdout.fold, spec.name, family) not in done
                ]
                if families:
                    self._screen_vectorizer(holdout, spec, families, config)
        self.progress.finish()
        return ScreeningReport(
            development_notes=len(notes),
            development_patients=len({note.subject_id for note in notes}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )

    def _screen_vectorizer(
        self,
        holdout: Holdout,
        spec: VectorizerSpec,
        families: Sequence[ClassifierFamily],
        config: ClassicalScreeningConfig,
    ) -> None:
        train, test = self.featurizer.fit_transform(
            spec,
            [note.text for note in holdout.train],
            [note.text for note in holdout.test],
        )
        subjects = [note.subject_id for note in holdout.train]
        for family in families:
            rows = []
            for diagnosis in Diagnosis:
                classifier = self.tuner.fit(
                    family,
                    train,
                    [note.has_label(diagnosis) for note in holdout.train],
                    subjects,
                    config.threshold_folds,
                    config.seed,
                )
                classifications = classifier.classify(test)
                truth = tuple(note.has_label(diagnosis) for note in holdout.test)
                metrics = self.scorer.score(
                    truth,
                    [item.predicted for item in classifications],
                    [item.score for item in classifications],
                )
                rows.append(
                    ScreeningResult(
                        fold=holdout.fold,
                        vectorizer=spec.name,
                        family=family,
                        diagnosis=diagnosis,
                        note_ids=tuple(note.note_id for note in holdout.test),
                        labels=truth,
                        classifications=classifications,
                        threshold=classifier.threshold,
                        converged=classifier.converged,
                        metrics=metrics,
                    )
                )
            self.progress.write(rows)


def _development_holdouts(
    notes: Sequence[InputNote],
    config: ClassicalScreeningConfig,
    partitioner: PatientPartitioner,
) -> tuple[Holdout, ...]:
    holdouts = patient_holdouts(notes, config.folds, config.seed, partitioner)
    for holdout in holdouts:
        train_subjects = [note.subject_id for note in holdout.train]
        for diagnosis in Diagnosis:
            ensure_patients_per_class(
                [note.has_label(diagnosis) for note in holdout.train],
                train_subjects,
                diagnosis,
                config.threshold_folds,
            )
    return holdouts
