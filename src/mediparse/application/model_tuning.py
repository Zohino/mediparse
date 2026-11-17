"""Společná smyčka ladění nad trénovací částí jednoho foldu evaluace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.patient_cv import patient_holdouts
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import TuningOutcome, TuningReport, macro_max_f1

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.application.ports.experiment import (
        InputTable,
        PatientPartitioner,
        SplitSource,
    )
    from mediparse.domain.matrix import MatrixRow
    from mediparse.domain.split import Holdout
    from mediparse.domain.tuning import (
        ParamValue,
        Trial,
        TuningConfig,
        ValidationScores,
    )


class ParameterSearch(Protocol):
    """Návrhy parametrů a strategie prořezávání bez vlastnictví rozpočtu."""

    def ask(self) -> Trial:
        """Navrhne parametry nového pokusu."""

    def should_prune(self, trial: Trial, value: float, step: int) -> bool:
        """Zapíše průběžný průměr a rozhodne o prořezání."""

    def complete(self, trial: Trial, value: float) -> None:
        """Uzavře dokončený pokus s cílovou hodnotou."""

    def prune(self, trial: Trial) -> None:
        """Uzavře prořezaný pokus započítaný do rozpočtu."""

    def best(self) -> tuple[Mapping[str, ParamValue], float]:
        """Vrátí parametry a cíl nejlepšího dokončeného pokusu."""


class ValidationScorer(Protocol):
    """Rodina modelu poskytuje pouze skóre validačních zpráv."""

    @property
    def parameters(self) -> frozenset[str]:
        """Jména všech parametrů rodiny včetně pevných voleb."""

    def score(
        self, params: Mapping[str, ParamValue], fold: Holdout
    ) -> ValidationScores:
        """Fitne rodinu na tréninku a vrátí skóre pěti diagnóz na validaci."""


@dataclass(frozen=True)
class ModelTuning:
    """Vyčerpá rozpočet řádku stejným protokolem pro každou rodinu."""

    table: InputTable
    split: SplitSource
    partitioner: PatientPartitioner
    search: ParameterSearch
    scorer: ValidationScorer

    def run(self, config: TuningConfig, row: MatrixRow) -> TuningReport:
        """Vybere výhradně vnější trénink a vyhodnotí pokusy ve sdílených foldech.

        Returns:
            Nejlepší parametry, stavy pokusů a agregovaná provenance tréninku.

        Raises:
            InvalidInputError: Řádek nebo prostor nesedí, případně nelze sestavit CV.
        """
        if row.id != config.row_id:
            msg = "Řádek matice neodpovídá row_id configu ladění."
            raise InvalidInputError(msg)
        if config.space.names != self.scorer.parameters:
            msg = "Jména prostoru ladění neodpovídají parametrům scoreru."
            raise InvalidInputError(msg)
        shared = self.split.read()
        holdout = shared.split.holdout(self.table.read(), config.fold)
        folds = patient_holdouts(
            holdout.train, config.inner_folds, config.seed, self.partitioner
        )
        completed, pruned, nonconverged = 0, 0, 0
        for _ in range(row.tuning_trials):
            was_pruned, converged = self._run_trial(self.search.ask(), folds)
            pruned += was_pruned
            completed += not was_pruned
            nonconverged += not converged
        best, value = self.search.best()
        return TuningReport(
            TuningOutcome(best, value, completed, pruned, nonconverged),
            train_notes=len(holdout.train),
            train_patients=len({note.subject_id for note in holdout.train}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )

    def _run_trial(self, trial: Trial, folds: Sequence[Holdout]) -> tuple[bool, bool]:
        values: list[float] = []
        converged = True
        for step, fold in enumerate(folds):
            scores = self.scorer.score(trial.params, fold)
            values.append(macro_max_f1(fold.test, scores))
            converged &= scores.converged
            if step + 1 < len(folds) and self.search.should_prune(
                trial, sum(values) / len(values), step
            ):
                self.search.prune(trial)
                return True, converged
        self.search.complete(trial, sum(values) / len(values))
        return False, converged
