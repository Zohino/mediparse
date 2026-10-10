"""Vstupní bod kroku Snakemake ``evaluate_smoketest_model``: metriky z predikcí.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Pracuje jen s tabulkou predikcí, proto smí běžet kdekoli.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.model_evaluation import ModelEvaluation
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.metrics_table import ParquetMetricsTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_metrics import SklearnScorer

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(*, predictions: Path, metrics: Path, log: Path) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(predictions=predictions, metrics=metrics)


@refusing_invalid_input
def run(*, predictions: Path, metrics: Path) -> ExitCode:
    """Složí use case z tabulky predikcí a adaptérů scikit-learn a spočte metriky.

    Returns:
        OK po zápisu metrik; REFUSED pro chybějící nebo neplatné predikce.
    """
    step = ModelEvaluation(
        predictions=ParquetPredictionTable(predictions),
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
