"""Společná smyčka ladění nad trénovací částí jednoho foldu evaluace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.patient_cv import patient_holdouts
from mediparse.application.ports.experiment import ParameterSearch, TrialScorer
from mediparse.application.tuning_trial import run_trial
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import TuningOutcome, TuningReport

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.application.ports.experiment import (
        InputTable,
        PatientPartitioner,
        SplitSource,
    )
    from mediparse.domain.matrix import MatrixRow
    from mediparse.domain.tuning import (
        ParamValue,
        TuningConfig,
    )


class TuningSearch(ParameterSearch, Protocol):
    """Hledání jednoho modelu s výběrem nejlepšího pokusu."""

    def best(self) -> tuple[Mapping[str, ParamValue], float]:
        """Vrátí parametry a cíl nejlepšího dokončeného pokusu."""


class ValidationScorer(TrialScorer, Protocol):
    """Rodina modelu poskytuje pouze skóre validačních zpráv."""

    @property
    def parameters(self) -> frozenset[str]:
        """Jména všech parametrů rodiny včetně pevných voleb."""


@dataclass(frozen=True)
class ModelTuning:
    """Vyčerpá rozpočet řádku stejným protokolem pro každou rodinu."""

    table: InputTable
    split: SplitSource
    partitioner: PatientPartitioner
    search: TuningSearch
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
            result = run_trial(self.search, self.scorer, self.search.ask(), folds)
            pruned += result.pruned
            completed += not result.pruned
            nonconverged += not result.converged
        best, value = self.search.best()
        return TuningReport(
            TuningOutcome(best, value, completed, pruned, nonconverged),
            train_notes=len(holdout.train),
            train_patients=len({note.subject_id for note in holdout.train}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )
