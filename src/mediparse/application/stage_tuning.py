"""Jedno ladění CASH nad pacientským podvzorkem vývojové části."""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.patient_cv import (
    development_notes,
    development_sample,
    patient_holdouts,
)
from mediparse.application.tuning_trial import (
    TrialRun,
    score_trial,
    trial_results,
    write_trial,
)
from mediparse.domain.stage_tuning import (
    BRANCH_KEY,
    StageTrialRecord,
    StageTuningReport,
    best_trial,
)
from mediparse.domain.tuning import Trial

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from mediparse.application.ports.experiment import (
        Clock,
        PatientNoteTable,
        PatientPartitioner,
        SplitSource,
        StageProgress,
        StageSearch,
        TrialScorer,
    )
    from mediparse.domain.split import Holdout
    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue


logger = logging.getLogger(__name__)

type RunningTrials = dict[Future[tuple[TrialRun, float]], tuple[Trial, bool]]


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
        shared, notes = development_notes(self.table, self.split)
        sample = development_sample(
            notes, config.subsample_parts, config.seed, self.partitioner
        )
        folds = patient_holdouts(sample, config.folds, config.seed, self.partitioner)
        records = list(self.progress.completed())
        self.search.restore([(Trial(r.number, r.params), r.value) for r in records])
        self._run_trials(config, folds, records)
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

    def _run_trials(
        self,
        config: StageTuningConfig,
        folds: Sequence[Holdout],
        records: list[StageTrialRecord],
    ) -> None:
        # Pád může ztratit běžící výchozí bod i po dokončení jiných větví.
        completed_defaults = {r.params[BRANCH_KEY] for r in records if r.default}
        defaults = iter(
            b.default_params()
            for b in config.branches
            if b.name not in completed_defaults
        )
        running: RunningTrials = {}
        errors: dict[int, Exception] = {}
        with ThreadPoolExecutor(max_workers=config.concurrent_trials) as pool:
            while len(records) < config.trials:
                if not errors:
                    self._fill_trials(
                        pool,
                        config,
                        folds,
                        records=records,
                        defaults=defaults,
                        running=running,
                    )
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                self._collect_trials(
                    [future for future in running if future in done],
                    running,
                    folds,
                    records,
                    errors,
                )
        if errors:
            # Ostatní GPU necháme dokončit a zapsat, aby se hotová práce neztratila.
            first, *others = sorted(errors)
            for number in others:
                logger.error(
                    "Pokus %d selhal: %s", number, type(errors[number]).__name__
                )
            raise errors[first]

    # Rozpočet, foldy, výchozí body a běžící práce jsou nezávislé vstupy plánování.
    def _fill_trials(  # ruff: ignore[too-many-arguments]
        self,
        pool: ThreadPoolExecutor,
        config: StageTuningConfig,
        folds: Sequence[Holdout],
        *,
        records: Sequence[StageTrialRecord],
        defaults: Iterator[Mapping[str, ParamValue]],
        running: RunningTrials,
    ) -> None:
        while (
            len(running) < config.concurrent_trials
            and len(records) + len(running) < config.trials
        ):
            params = next(defaults, None)
            if params is not None:
                self.search.enqueue(params)
            trial = self.search.ask()
            running[pool.submit(self._timed_run, trial.params, folds)] = (
                trial,
                params is not None,
            )

    def _collect_trials(
        self,
        done: Sequence[Future[tuple[TrialRun, float]]],
        running: RunningTrials,
        folds: Sequence[Holdout],
        records: list[StageTrialRecord],
        errors: dict[int, Exception],
    ) -> None:
        for future in done:
            trial, default = running.pop(future)
            try:
                run, seconds = future.result()
            # Každá chyba skórování musí počkat na zápis ostatních GPU a znovu se vyhodit.
            except Exception as error:  # ruff: ignore[blind-except]
                errors[trial.number] = error
                continue
            self.search.complete(trial, run.value)
            # Čísla Optuny mají po pádu mezery; checkpoint čísluje pořadí zápisu.
            record = StageTrialRecord(
                len(records), trial.params, default, run.value, run.converged, seconds
            )
            write_trial(self.progress, record, trial_results(run, folds))
            records.append(record)

    def _timed_run(
        self,
        params: Mapping[str, ParamValue],
        folds: Sequence[Holdout],
    ) -> tuple[TrialRun, float]:
        start = self.clock.now()
        run = score_trial(self.scorer, params, folds)
        return run, self.clock.now() - start
