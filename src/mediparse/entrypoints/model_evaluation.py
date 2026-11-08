"""Vstupní bod kroku Snakemake ``evaluate_model``: metriky z predikcí.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Čte i zprávy a split, aby ověřil úplnost predikcí a odvodil pravdu z labelů;
u reálné kohorty proto běží tam, kde jsou data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.model_evaluation import ModelEvaluation
from mediparse.domain.environment import Environment
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import EvaluationManifest, ensure_training_lineage
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.metrics_table import ParquetMetricsTable
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.run_manifest_file import load_run_manifest
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.split_table import ParquetSplitTable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.domain.environment import SourceRevision

logger = logging.getLogger(__name__)

PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "pydantic",
    "snakemake",
)


def main(
    *,
    predictions: Path,
    cohort: Path,
    split: Path,
    training_manifest: Path,
    metrics: Path,
    manifest: Path,
    log: Path,
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(
        environ=os.environ,
        predictions=predictions,
        cohort=cohort,
        split=split,
        training_manifest=training_manifest,
        metrics=metrics,
        manifest=manifest,
        revision=revision_source(os.environ, Path.cwd()),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    predictions: Path,
    cohort: Path,
    split: Path,
    training_manifest: Path,
    metrics: Path,
    manifest: Path,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    """Složí use case z tabulek predikcí, zpráv, splitu a adaptérů scikit-learn.

    Returns:
        OK po zápisu metrik a manifestu; REFUSED pro chybějící či neplatné vstupy,
        nečitelnou revizi, porušenou návaznost na trénink, neúplné predikce
        a pro neoznačená data v prostředí, které nesmí číst MIMIC.

    Raises:
        InvalidInputError: Vstupy neodpovídají tréninku; dekorátor přeloží na REFUSED.
    """
    check_inputs_access(environ, (predictions, cohort, split))
    training = load_run_manifest(training_manifest)
    environment = Environment(packages=installed_versions(PACKAGES), source=revision())
    predictions_digest = file_sha256(predictions)
    dataset_digest = file_sha256(cohort)
    split_digest = ParquetSplitTable(split).read().split.sha256()
    try:
        ensure_training_lineage(
            training,
            predictions_sha256=predictions_digest,
            dataset_sha256=dataset_digest,
            split_sha256=split_digest,
        )
    except InvalidInputError as error:
        msg = f"Manifest {training_manifest}: {error}"
        raise InvalidInputError(msg) from error
    step = ModelEvaluation(
        predictions=ParquetPredictionTable(predictions),
        table=ParquetNoteTable(cohort),
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
    write_model(
        manifest,
        EvaluationManifest(
            row_id=training.row_id,
            training_manifest_sha256=file_sha256(training_manifest),
            predictions_sha256=predictions_digest,
            dataset_sha256=dataset_digest,
            split_sha256=split_digest,
            metrics_sha256=file_sha256(metrics),
            environment=environment,
        ),
    )
    return ExitCode.OK
