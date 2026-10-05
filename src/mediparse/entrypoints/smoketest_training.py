"""Vstupní bod kroku Snakemake ``train_smoketest_model``: model a held-out metriky.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Pracuje jen se vstupní tabulkou smoketestu, proto smí běžet kdekoli.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.smoketest_training import SmoketestTraining
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.report_file import JsonReportFile
from mediparse.infrastructure.sklearn_classifier import (
    GroupedHoldout,
    LinearSvmTrainer,
    SklearnScorer,
    SkopsModelFile,
)
from mediparse.infrastructure.training_config import load_training_config

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *, notes: Path, config: Path, model: Path, metrics: Path, log: Path
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(notes=notes, config=config, model=model, metrics=metrics)


@refusing_invalid_input
def run(*, notes: Path, config: Path, model: Path, metrics: Path) -> ExitCode:
    """Složí use case z tabulky, configu a adaptérů scikit-learn a spustí trénink.

    Returns:
        OK po uložení modelu a metrik; REFUSED pro chybějící nebo neplatnou
        tabulku a config.
    """
    training = load_training_config(config)
    step = SmoketestTraining(
        table=ParquetNoteTable(notes),
        splitter=GroupedHoldout(training.folds, training.seed),
        trainer=LinearSvmTrainer(training.regularization, training.seed),
        scorer=SklearnScorer(),
        store=SkopsModelFile(model),
        report=JsonReportFile(metrics),
    )
    report = step.run(training)
    logger.info(
        "Model diagnózy %s: F1 %.3f na %d zprávách, uložen do %s.",
        report.diagnosis,
        report.metrics.f1,
        report.test_notes,
        model,
    )
    return ExitCode.OK
