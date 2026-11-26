"""Kontrola rodin a doladění vítězné větve na celé vývojové části."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.patient_cv import patient_holdouts
from mediparse.application.tuning_trial import score_trial, trial_results
from mediparse.domain.stage_refinement import StageRefinementReport, branch_config
from mediparse.domain.stage_tuning import (
    BRANCH_KEY,
    StageTrialRecord,
    best_trial,
    family_candidates,
)
from mediparse.domain.tuning import Trial

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import (
        Clock,
        PatientNoteTable,
        PatientPartitioner,
        SplitSource,
        StageProgress,
        StageSearches,
        TrialScorer,
    )
    from mediparse.domain.stage_refinement import RefinementConfig
    from mediparse.domain.stage_tuning import StageTuningConfig


@dataclass(frozen=True)
class StageRefinement:
    """Vyhodnotí dvě rodiny a vyčerpá společný rozpočet vítězné větve."""

    table: PatientNoteTable
    split: SplitSource
    partitioner: PatientPartitioner
    searches: StageSearches
    scorer: TrialScorer
    progress: StageProgress
    clock: Clock

    def run(
        self,
        stage: StageTuningConfig,
        refinement: RefinementConfig,
        screening: Sequence[StageTrialRecord],
        trials: int,
    ) -> StageRefinementReport:
        """Obnoví kontrolu i doladění nad stejnými pacientskými foldy.

        Returns:
            Vítěz a agregáty kontroly a doladění s původem splitu.
        """
        candidates = family_candidates(screening)
        studies = {
            candidate.params[BRANCH_KEY]: branch_config(
                stage,
                str(candidate.params[BRANCH_KEY]),
                trials=trials,
                random_startup_trials=refinement.random_startup_trials,
            )
            for candidate in candidates
        }
        shared = self.split.read()
        shared.split.ensure_covers(self.table.subjects())
        notes = self.table.read_patients(shared.split.development)
        folds = patient_holdouts(notes, stage.folds, stage.seed, self.partitioner)
        records = list(self.progress.completed())
        for candidate in candidates[len(records) :]:
            start = self.clock.now()
            run = score_trial(self.scorer, candidate.params, folds)
            record = StageTrialRecord(
                len(records),
                candidate.params,
                default=True,
                value=run.value,
                converged=run.converged,
                seconds=self.clock.now() - start,
            )
            self.progress.write(record, trial_results(run, folds))
            records.append(record)
        winner = best_trial(records[:2])
        study = studies[winner.params[BRANCH_KEY]]
        search = self.searches.create(study)
        search.restore([
            (Trial(r.number, r.params), r.value) for r in (winner, *records[2:])
        ])
        for _ in range(len(records) - 1, trials):
            start = self.clock.now()
            trial = search.ask()
            run = score_trial(self.scorer, trial.params, folds)
            search.complete(trial, run.value)
            record = StageTrialRecord(
                len(records),
                trial.params,
                default=False,
                value=run.value,
                converged=run.converged,
                seconds=self.clock.now() - start,
            )
            self.progress.write(record, trial_results(run, folds))
            records.append(record)
        self.progress.finish()
        return StageRefinementReport(
            study=study,
            checks=(records[0], records[1]),
            best=best_trial(records),
            completed_trials=len(records),
            nonconverged_trials=sum(not r.converged for r in records),
            seconds=sum(r.seconds for r in records),
            development_notes=len(notes),
            development_patients=len({n.subject_id for n in notes}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
        )
