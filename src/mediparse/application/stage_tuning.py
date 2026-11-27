"""Jedno ladění CASH nad pacientským podvzorkem vývojové části."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.patient_cv import patient_holdouts
from mediparse.application.tuning_trial import score_trial, trial_results, write_trial
from mediparse.domain.stage_tuning import (
    StageTrialRecord,
    StageTuningReport,
    best_trial,
)
from mediparse.domain.tuning import Trial

if TYPE_CHECKING:
    from mediparse.application.ports.experiment import (
        Clock,
        PatientNoteTable,
        PatientPartitioner,
        SplitSource,
        StageProgress,
        StageSearch,
        TrialScorer,
    )
    from mediparse.domain.stage_tuning import StageTuningConfig


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
            run = score_trial(self.scorer, trial.params, folds)
            self.search.complete(trial, run.value)
            record = StageTrialRecord(
                trial.number,
                trial.params,
                default,
                run.value,
                run.converged,
                self.clock.now() - start,
            )
            write_trial(self.progress, record, trial_results(run, folds))
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
