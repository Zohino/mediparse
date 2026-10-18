"""Vstupní bod kroku Snakemake ``evaluate_smoketest_model``: metriky z predikcí.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Čte i zprávy a split, aby ověřil úplnost predikcí; u reálné kohorty proto
běží tam, kde jsou data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.model_evaluation import ModelEvaluation
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.metrics_table import ParquetMetricsTable
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.split_table import ParquetSplitTable

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *, predictions: Path, notes: Path, split: Path, metrics: Path, log: Path
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(predictions=predictions, notes=notes, split=split, metrics=metrics)


@refusing_invalid_input
def run(*, predictions: Path, notes: Path, split: Path, metrics: Path) -> ExitCode:
    """Složí use case z tabulek predikcí, zpráv, splitu a adaptérů scikit-learn.

    Returns:
        OK po zápisu metrik; REFUSED pro chybějící či neplatné vstupy a neúplné predikce.
    """
    step = ModelEvaluation(
        predictions=ParquetPredictionTable(predictions),
        table=ParquetNoteTable(notes),
        split=ParquetSplitTable(split),
        scorer=SklearnScorer(),
        metrics=ParquetMetricsTable(metrics),
    )
    for item in step.run():
        logger.info(
            "Řádek %s, diagnóza %s: F1 %.3f, ROC-AUC %.3f, PR-AUC %.3f na %d zprávách.",
            item.row_id,
            item.diagnosis,
            item.metrics.f1,
            item.metrics.roc_auc,
            item.metrics.pr_auc,
            item.test_notes,
        )
    return ExitCode.OK
