"""Jedno ladění CASH nad pacientským podvzorkem vývojové části."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.patient_cv import patient_holdouts
from mediparse.application.ports.experiment import ParameterSearch
from mediparse.application.tuning_trial import run_trial
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.stage_tuning import (
    StageFoldResult,
    StageTrialRecord,
    StageTuningReport,
    best_trial,
)
from mediparse.domain.tuning import Trial

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.application.ports.experiment import (
        Clock,
        PatientNoteTable,
        PatientPartitioner,
        SplitSource,
        TrialScorer,
    )
    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue


class StageSearch(ParameterSearch, Protocol):
    """Návrhy větví s výchozími body a obnovou historie."""

    def enqueue(self, params: Mapping[str, ParamValue]) -> None:
        """Vloží parametry následujícího pokusu."""

    def restore(self, trials: Sequence[tuple[Trial, float]]) -> None:
        """Obnoví dokončené pokusy v pořadí čísel."""


class StageProgress(Protocol):
    """Trvalé pokusy a jejich validační predikce."""

    def completed(self) -> tuple[StageTrialRecord, ...]:
        """Ověří klíč a vrátí dokončené pokusy."""

    def write(
        self, record: StageTrialRecord, results: Sequence[StageFoldResult]
    ) -> None:
        """Zapíše jeden pokus se všemi foldy a diagnózami."""

    def finish(self) -> None:
        """Složí výstupy a uklidí checkpoint."""


@dataclass(frozen=True)
class StageTuning:
    """Vyčerpá společný rozpočet na stratifikovaném vývojovém vzorku."""

    table: PatientNoteTable
    split: SplitSource
    partitioner: PatientPartitioner
    search: StageSearch
    scorer: TrialScorer
    progress: StageProgress
    clock: Clock

    def run(self, config: StageTuningConfig) -> StageTuningReport:
        """Obnoví hotové pokusy a dopočítá zbytek na stejných foldech.

        Returns:
            Vítěz a velikosti vzorku s provenance původního splitu.

        Raises:
            InvalidInputError: Hledání prořezalo pokus stupně.
        """
        shared = self.split.read()
        shared.split.ensure_covers(self.table.subjects())
        notes = self.table.read_patients(shared.split.development)
        sample = (
            notes
            if config.subsample_parts == 1
            else patient_holdouts(
                notes, config.subsample_parts, config.seed, self.partitioner
            )[0].test
        )
        folds = patient_holdouts(sample, config.folds, config.seed, self.partitioner)
        records = list(self.progress.completed())
        self.search.restore([(Trial(r.number, r.params), r.value) for r in records])
        for n in range(len(records), config.trials):
            default = n < len(config.branches)
            if default:
                self.search.enqueue(config.branches[n].default_params())
            start = self.clock.now()
            trial = self.search.ask()
            run = run_trial(self.search, self.scorer, trial, folds)
            if run.pruned:
                msg = "Ladění stupně neprořezává."
                raise InvalidInputError(msg)
            results = [
                StageFoldResult(
                    fold=fold.fold,
                    diagnosis=diagnosis,
                    note_ids=tuple(note.note_id for note in fold.test),
                    labels=tuple(note.has_label(diagnosis) for note in fold.test),
                    scores=run.scores[i].scores[diagnosis],
                    max_f1=run.f1[i][diagnosis],
                    converged=run.scores[i].converged,
                    features=run.scores[i].features,
                )
                for i, fold in enumerate(folds)
                for diagnosis in Diagnosis
            ]
            record = StageTrialRecord(
                trial.number,
                trial.params,
                default,
                run.value,
                run.converged,
                self.clock.now() - start,
            )
            self.progress.write(record, results)
            records.append(record)
        self.progress.finish()
        return StageTuningReport(
            best=best_trial(records),
            completed_trials=len(records),
            seconds=sum(r.seconds for r in records),
            nonconverged_trials=sum(not r.converged for r in records),
            sample_notes=len(sample),
            sample_patients=len({note.subject_id for note in sample}),
            development_notes=len(notes),
            development_patients=len({note.subject_id for note in notes}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )
