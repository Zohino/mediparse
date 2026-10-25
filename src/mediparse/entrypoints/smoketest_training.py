"""Vstupní bod kroku Snakemake ``train_smoketest_model``: model a predikce odložených zpráv.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Pracuje jen se vstupní tabulkou smoketestu, proto smí běžet kdekoli.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.smoketest_training import SmoketestTraining
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import RunManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.training_input import training_input
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
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
    from collections.abc import Callable

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
    notes: Path,
    config: Path,
    lexicon: Path,
    split: Path,
    model: Path,
    predictions: Path,
    manifest: Path,
    log: Path,
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(
        notes=notes,
        config=config,
        lexicon=lexicon,
        split=split,
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=revision_source(os.environ, Path.cwd()),
    )


@refusing_invalid_input
def run(
    *,
    notes: Path,
    config: Path,
    lexicon: Path,
    split: Path,
    model: Path,
    predictions: Path,
    manifest: Path,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    """Složí use case z tabulky, configu, souboru splitu a adaptérů scikit-learn a spustí trénink.

    Returns:
        OK po uložení modelu, predikcí a manifestu; REFUSED pro chybějící nebo
        neplatnou tabulku, config, lexikon a soubor splitu a pro nečitelnou revizi zdroje.
    """
    training = load_training_config(config)
    source = revision()
    table, strip = training_input(notes, training, lexicon)
    step = SmoketestTraining(
        table=table,
        split=ParquetSplitTable(split),
        trainer=LinearSvmTrainer(
            training.regularization, training.seed, normalize_clinical_text=False
        ),
        store=SkopsModelFile(model),
        predictions=ParquetPredictionTable(predictions),
    )
    report = step.run(training)
    write_model(
        manifest,
        RunManifest(
            row_id=training.row_id,
            dataset_sha256=file_sha256(notes),
            config_sha256=file_sha256(config),
            seed=training.seed,
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            diagnostic_strip=strip,
            model=ESTIMATOR,
            model_revision=None,
            tokenizer_revision=None,
            environment=Environment(
                packages=installed_versions(PACKAGES), source=source
            ),
        ),
    )
    logger.info(
        "Model diagnózy %s: %d trénovacích a %d testovacích zpráv, uložen do %s.",
        report.diagnosis,
        report.train_notes,
        report.test_notes,
        model,
    )
    return ExitCode.OK
