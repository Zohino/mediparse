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
from mediparse.domain.run_manifest import RunManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.run_manifest_file import JsonRunManifestFile
from mediparse.infrastructure.sklearn_classifier import (
    ESTIMATOR,
    GroupedHoldout,
    LinearSvmTrainer,
    SkopsModelFile,
)
from mediparse.infrastructure.source_revision import (
    COMMIT_VARIABLE,
    EnvironmentRevision,
    GitRevision,
)
from mediparse.infrastructure.training_config import load_training_config

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.domain.run_manifest import SourceRevision

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


def revision_source(
    environ: Mapping[str, str], root: Path
) -> Callable[[], SourceRevision]:
    """Vybere zdroj revize: prostředí image, má-li MEDIPARSE_COMMIT, jinak git.

    Args:
        environ: Proměnné prostředí procesu.
        root: Kořen repozitáře pro čtení z gitu.

    Returns:
        Funkce, která přečte revizi zdroje.
    """
    if COMMIT_VARIABLE in environ:
        return EnvironmentRevision(environ).read
    return GitRevision(root).read


def main(
    *,
    notes: Path,
    config: Path,
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
    model: Path,
    predictions: Path,
    manifest: Path,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    """Složí use case z tabulky, configu a adaptérů scikit-learn a spustí trénink.

    Returns:
        OK po uložení modelu, predikcí a manifestu; REFUSED pro chybějící nebo
        neplatnou tabulku a config a pro nečitelnou revizi zdroje.
    """
    training = load_training_config(config)
    source = revision()
    step = SmoketestTraining(
        table=ParquetNoteTable(notes),
        splitter=GroupedHoldout(training.folds, training.seed),
        trainer=LinearSvmTrainer(training.regularization, training.seed),
        store=SkopsModelFile(model),
        predictions=ParquetPredictionTable(predictions),
    )
    report = step.run(training)
    JsonRunManifestFile(manifest).write(
        RunManifest(
            row_id=training.row_id,
            dataset_sha256=file_sha256(notes),
            config_sha256=file_sha256(config),
            seed=training.seed,
            model=ESTIMATOR,
            model_revision=None,
            tokenizer_revision=None,
            packages=installed_versions(PACKAGES),
            source=source,
        )
    )
    logger.info(
        "Model diagnózy %s: %d trénovacích a %d testovacích zpráv, uložen do %s.",
        report.diagnosis,
        report.train_notes,
        report.test_notes,
        model,
    )
    return ExitCode.OK
