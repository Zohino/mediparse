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
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import RunManifest, ensure_tuning_lineage
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_input_table import (
    model_input_table,
    text_normalization,
)
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.nvml_gpu import gpu_monitor
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.run_manifest_file import load_tuning_manifest
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
    from mediparse.domain.run_manifest import TuningManifest
    from mediparse.domain.training import TrainingConfig
    from mediparse.domain.tuning import ParamValue

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


def _parameters(
    config: TrainingConfig, tuning: TuningManifest | None
) -> dict[str, ParamValue]:
    """Vybere právě jeden zdroj parametrů tréninku.

    Returns:
        Nejlepší parametry manifestu, nebo pevné parametry configu.

    Raises:
        InvalidInputError: Parametry chybějí, nebo jsou přítomné oba zdroje.
    """
    if tuning is None and config.parameters is not None:
        return config.parameters
    if tuning is not None and config.parameters is None:
        return tuning.best
    msg = "Trénink vyžaduje právě jeden zdroj parametrů: tuning nebo config.parameters."
    raise InvalidInputError(msg)


def main(
    *,
    cohort: Path,
    config: Path,
    candidates: Path,
    lexicon: Path,
    split: Path,
    tuning: Path | None,
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
        tuning=tuning,
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
    tuning: Path | None,
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
        a pro neoznačená data v prostředí, které nesmí číst MIMIC. Odmítne také chybějící
        či dvojitý zdroj parametrů a manifest ladění z jiného tréninku.
    """
    check_inputs_access(environ, (cohort, split))
    training = load_training_config(config)
    tuned = load_tuning_manifest(tuning) if tuning is not None else None
    parameters = _parameters(training, tuned)
    source = revision()
    table, strip = model_input_table(
        cohort=cohort, model_input=training.input, lexicon=lexicon
    )
    normalization = text_normalization(candidates, training.input)
    dataset_sha256 = file_sha256(cohort)
    shared_split = ParquetSplitTable(split)
    if tuned is not None:
        ensure_tuning_lineage(
            tuned,
            row_id=training.row_id,
            fold=training.fold,
            dataset_sha256=dataset_sha256,
            split_sha256=shared_split.read().split.sha256(),
            diagnostic_strip=strip,
            text_preprocessing=training.input.text_preprocessing,
            text_normalization=normalization,
        )
    step = ModelTraining(
        table=table,
        split=shared_split,
        trainer=LinearSvmTrainer(
            parameters,
            training.seed,
            normalization=normalization,
            threshold_folds=training.threshold_folds,
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
            dataset_sha256=dataset_sha256,
            predictions_sha256=file_sha256(predictions),
            config_sha256=file_sha256(config),
            tuning_manifest_sha256=file_sha256(tuning) if tuning is not None else None,
            parameters=parameters,
            threshold=report.threshold,
            threshold_folds=training.threshold_folds,
            converged=report.converged,
            seed=training.seed,
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            diagnostic_strip=strip,
            text_preprocessing=training.input.text_preprocessing,
            text_normalization=normalization,
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
