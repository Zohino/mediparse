"""Screening klasické mřížky s vnější CV jen mezi vývojovými pacienty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.classical_screening import ScreeningReport, ScreeningResult
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import ensure_min_cell
from mediparse.domain.split import Split, ensure_patients_per_class, strata

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


class ResultSink(Protocol):
    """Výstup buněk screeningu."""

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Zapíše všechny buňky mřížky po dokončení výpočtu."""


@dataclass(frozen=True)
class ClassicalScreening[F]:
    """Provede mřížku klasického stupně na vývojové části původního splitu."""

    table: PatientNoteTable
    split: SplitSource
    partitioner: PatientPartitioner
    featurizer: Featurizer[F]
    tuner: ThresholdTuner[F]
    scorer: Scorer
    results: ResultSink
    predictions: ResultSink

    def run(self, config: ClassicalScreeningConfig) -> ScreeningReport:
        """Ověří foldy před fitováním a zapíše spočtené buňky do obou výstupů.

        Args:
            config: Mřížka v pořadí výpočtu, parametry CV a seed screeningu.

        Returns:
            Velikost vývojové části a provenance původního splitu.

        Raises:
            InvalidInputError: Pacient tabulky je mimo split, vývojová část nemá dost
                pacientů v každé třídě, některý test foldu nesplňuje ``MIN_CELL``
                nebo trénink foldu nemá dost pacientů pro vnitřní CV prahu.
        """
        shared = self.split.read()
        shared.split.ensure_covers(self.table.subjects())
        notes = self.table.read_patients(shared.split.development)
        holdouts = _development_holdouts(notes, config, self.partitioner)
        rows: list[ScreeningResult] = []
        for holdout in holdouts:
            for spec in config.vectorizers:
                rows.extend(self._screen_vectorizer(holdout, spec, config))
        self.results.write(rows)
        self.predictions.write(rows)
        return ScreeningReport(
            development_notes=len(notes),
            development_patients=len({note.subject_id for note in notes}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )

    def _screen_vectorizer(
        self, holdout: Holdout, spec: VectorizerSpec, config: ClassicalScreeningConfig
    ) -> tuple[ScreeningResult, ...]:
        train, test = self.featurizer.fit_transform(
            spec,
            [note.text for note in holdout.train],
            [note.text for note in holdout.test],
        )
        subjects = [note.subject_id for note in holdout.train]
        rows = []
        for family in config.families:
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
        return tuple(rows)


def _development_holdouts(
    notes: Sequence[InputNote],
    config: ClassicalScreeningConfig,
    partitioner: PatientPartitioner,
) -> tuple[Holdout, ...]:
    subjects = [note.subject_id for note in notes]
    for diagnosis in Diagnosis:
        ensure_patients_per_class(
            [note.has_label(diagnosis) for note in notes],
            subjects,
            diagnosis,
            config.folds,
        )
    parts = partitioner.partition(
        subjects, strata(notes, config.folds), config.folds, config.seed
    )
    split = Split(parts, development=frozenset())
    holdouts = tuple(split.holdout(notes, fold) for fold in range(config.folds))
    for holdout in holdouts:
        train_subjects = [note.subject_id for note in holdout.train]
        for diagnosis in Diagnosis:
            ensure_min_cell(
                diagnosis, [note.has_label(diagnosis) for note in holdout.test]
            )
            ensure_patients_per_class(
                [note.has_label(diagnosis) for note in holdout.train],
                train_subjects,
                diagnosis,
                config.threshold_folds,
            )
    return holdouts
