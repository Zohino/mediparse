"""Pilot transformeru na foldu nula společného vzorku předkola."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.patient_cv import (
    development_notes,
    development_sample,
    patient_holdouts,
)
from mediparse.domain.split import patient_fraction
from mediparse.domain.transformer_pilot import (
    PilotRecord,
    TransformerPilotReport,
    best_pilot_epoch,
)
from mediparse.domain.tuning import fold_max_f1, macro_max_f1

if TYPE_CHECKING:
    from mediparse.application.ports.experiment import (
        Clock,
        PatientNoteTable,
        PatientPartitioner,
        PilotSink,
        PilotTrainer,
        SplitSource,
    )
    from mediparse.domain.transformer_pilot import TransformerPilotConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerPilot:
    """Změří varianty na vnořených pacientech se společnou validací."""

    table: PatientNoteTable
    split: SplitSource
    partitioner: PatientPartitioner
    trainer: PilotTrainer
    sink: PilotSink
    clock: Clock

    def run(self, config: TransformerPilotConfig) -> TransformerPilotReport:
        """Provede běhy v pořadí configu a po každém přepíše agregáty.

        Returns:
            Tabulka epoch a provenance společného vývojového vzorku.
        """
        start = self.clock.now()
        shared, notes = development_notes(self.table, self.split)
        sample = development_sample(
            notes, config.subsample_parts, config.seed, self.partitioner
        )
        fold0 = patient_holdouts(sample, config.folds, config.seed, self.partitioner)[0]
        records: list[PilotRecord] = []
        for run in config.runs:
            subset = patient_fraction(fold0.train, run.fraction, config.seed)
            holdout = patient_holdouts(
                subset, config.stopping_parts, config.seed, self.partitioner
            )[0]
            measured = [
                PilotRecord(
                    run=run.encoder.name,
                    model=run.encoder.model,
                    window=run.encoder.window,
                    aggregation=run.aggregation,
                    freeze_embeddings=run.freeze_embeddings,
                    fraction=run.fraction,
                    epoch=epoch.epoch,
                    fit_notes=len(holdout.train),
                    fit_patients=len({n.subject_id for n in holdout.train}),
                    stop_notes=len(holdout.test),
                    validation_notes=len(fold0.test),
                    seconds=epoch.seconds,
                    validation_seconds=epoch.validation_seconds,
                    peak_memory_bytes=epoch.peak_memory_bytes,
                    stop_loss=epoch.stop_loss,
                    macro_f1=macro_max_f1(fold_max_f1(fold0.test, epoch.scores)),
                )
                for epoch in self.trainer.train(
                    run,
                    config.training,
                    holdout.train,
                    holdout.test,
                    [n.text for n in fold0.test],
                )
            ]
            records.extend(measured)
            self.sink.write(tuple(records))
            best = best_pilot_epoch(measured)
            if best is None:
                # Bez konečné stop BCE nelze vybrat epochu; F1 by předstíralo vítěze.
                logger.info(
                    "Pilot %s: %d epoch, nekonvergoval, %.1f s.",
                    run.encoder.name,
                    len(measured),
                    sum(r.seconds for r in measured),
                )
                continue
            logger.info(
                "Pilot %s: %d epoch, macro-F1 %.4f, %.1f s.",
                run.encoder.name,
                len(measured),
                best.macro_f1,
                sum(r.seconds for r in measured),
            )
        return TransformerPilotReport(
            records=tuple(records),
            sample_notes=len(sample),
            sample_patients=len({n.subject_id for n in sample}),
            development_notes=len(notes),
            development_patients=len({n.subject_id for n in notes}),
            split_config=shared.config,
            split_sha256=shared.split.sha256(),
            seconds=self.clock.now() - start,
        )
