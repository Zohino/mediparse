"""Vstupní bod kroku Snakemake ``train_model``: model a predikce odložených zpráv.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Čte kohortu, proto běží tam, kde jsou data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.model_training import ModelTraining
from mediparse.domain.costs import TrainingCosts
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import RunManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.training_input import training_input
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.nvml_gpu import gpu_monitor
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import (
    ESTIMATOR,
    LinearSvmTrainer,
    SkopsModelFile,
)
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.split_table import ParquetSplitTable
from mediparse.infrastructure.training_config import load_training_config

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.domain.environment import SourceRevision

logger = logging.getLogger(__name__)

PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "skops",
    "pyarrow",
    "pydantic",
    "snakemake",
)


def main(
    *,
    cohort: Path,
    config: Path,
    candidates: Path,
    lexicon: Path,
    split: Path,
    model: Path,
    predictions: Path,
    manifest: Path,
    costs: Path,
    log: Path,
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(
        environ=os.environ,
        cohort=cohort,
        config=config,
        candidates=candidates,
        lexicon=lexicon,
        split=split,
        model=model,
        predictions=predictions,
        manifest=manifest,
        costs=costs,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        revision=revision_source(os.environ, Path.cwd()),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    cohort: Path,
    config: Path,
    candidates: Path,
    lexicon: Path,
    split: Path,
    model: Path,
    predictions: Path,
    manifest: Path,
    costs: Path,
    revision: Callable[[], SourceRevision],
    cuda_visible_devices: str | None,
    cuda_device_order: str | None,
) -> ExitCode:
    """Složí use case z tabulky, configu, souboru splitu a adaptérů scikit-learn a spustí trénink.

    Returns:
        OK po uložení modelu, predikcí, manifestu a nákladů; REFUSED pro chybějící nebo
        neplatnou tabulku, config, kandidáty (při zapnuté normalizaci), lexikon a soubor splitu,
        pro nečitelnou revizi zdroje, nezjistitelný HW stroje, selhání měření GPU
        a pro neoznačená data v prostředí, které nesmí číst MIMIC.
    """
    check_inputs_access(environ, (cohort, split))
    training = load_training_config(config)
    source = revision()
    table, strip = training_input(cohort=cohort, training=training, lexicon=lexicon)
    normalization = (
        load_text_normalization(
            candidates, variants=training.normalize_diagnosis_variants
        )
        if training.normalize_clinical_text
        else None
    )
    step = ModelTraining(
        table=table,
        split=ParquetSplitTable(split),
        trainer=LinearSvmTrainer(
            training.regularization, training.seed, normalization=normalization
        ),
        store=SkopsModelFile(model),
        predictions=ParquetPredictionTable(predictions),
    )
    with gpu_monitor(cuda_visible_devices, cuda_device_order) as monitor:
        hardware = machine_hardware(monitor.devices)
        report = step.run(training)
    write_model(
        manifest,
        RunManifest(
            row_id=training.row_id,
            dataset_sha256=file_sha256(cohort),
            predictions_sha256=file_sha256(predictions),
            config_sha256=file_sha256(config),
            seed=training.seed,
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            diagnostic_strip=strip,
            normalize_clinical_text=normalization,
            model=ESTIMATOR,
            model_revision=None,
            tokenizer_revision=None,
            hardware=hardware,
            environment=Environment(
                packages=installed_versions(PACKAGES), source=source
            ),
        ),
    )
    write_model(costs, TrainingCosts(gpu=monitor.measurement()))
    logger.info(
        "Model diagnózy %s: %d trénovacích a %d testovacích zpráv, uložen do %s.",
        report.diagnosis,
        report.train_notes,
        report.test_notes,
        model,
    )
    return ExitCode.OK
