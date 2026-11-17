"""Vstupní bod kroku tune_model: obecné ladění a publikovatelný manifest."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.model_tuning import ModelTuning
from mediparse.domain.environment import Environment
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.matrix import ModelName
from mediparse.domain.run_manifest import TuningManifest
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
from mediparse.infrastructure.matrix_manifest import load_matrix
from mediparse.infrastructure.optuna_search import PRUNER, SAMPLER, OptunaSearch
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.sklearn_tuning import SklearnLinearSvmScorer
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.split_table import ParquetSplitTable
from mediparse.infrastructure.tuning_config import load_tuning_config

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.application.model_tuning import ValidationScorer
    from mediparse.application.ports.experiment import Clock
    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.seed import Seed

logger = logging.getLogger(__name__)
PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "optuna",
    "pyarrow",
    "pydantic",
    "snakemake",
)


@dataclass(frozen=True, kw_only=True)
class TuningPaths:
    """Vstupní a výstupní cesty ladění modelu."""

    cohort: Path
    config: Path
    matrix: Path
    candidates: Path
    lexicon: Path
    split: Path
    storage: Path
    manifest: Path


def main(*, paths: TuningPaths, log: Path) -> ExitCode:
    """Převezme cesty pravidla a nastaví souborovou diagnostiku.

    Returns:
        Návratový kód ladění.
    """
    configure_logging(log)
    return run(
        environ=os.environ,
        paths=paths,
        revision=revision_source(os.environ, Path.cwd()),
        clock=PerfCounterClock(),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    paths: TuningPaths,
    revision: Callable[[], SourceRevision],
    clock: Clock,
) -> ExitCode:
    """Složí adaptéry pro řádek matice a změří celý use case ladění.

    Returns:
        OK po zápisu manifestu, REFUSED pro neplatné vstupy a nepovolený přístup k datům.
    """
    check_inputs_access(environ, (paths.cohort, paths.split))
    tuning = load_tuning_config(paths.config)
    row = load_matrix(paths.matrix).row(tuning.row_id)
    source = revision()
    table, strip = model_input_table(
        cohort=paths.cohort, model_input=row.input, lexicon=paths.lexicon
    )
    normalization = text_normalization(paths.candidates, row.input)
    scorer = _validation_scorer(row.model, normalization, tuning.seed)
    hardware = machine_hardware(())
    step = ModelTuning(
        table=table,
        split=ParquetSplitTable(paths.split),
        partitioner=StratifiedGroupFolds(),
        search=OptunaSearch.create(
            tuning.space, tuning.seed, tuning.pruner_startup_trials, paths.storage
        ),
        scorer=scorer,
    )
    started = clock.now()
    report = step.run(tuning, row)
    seconds = clock.now() - started
    outcome = report.outcome
    write_model(
        paths.manifest,
        TuningManifest(
            row_id=row.id,
            fold=tuning.fold,
            dataset_sha256=file_sha256(paths.cohort),
            config_sha256=file_sha256(paths.config),
            matrix_sha256=file_sha256(paths.matrix),
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            seed=tuning.seed,
            tuning_trials=row.tuning_trials,
            completed_trials=outcome.completed_trials,
            pruned_trials=outcome.pruned_trials,
            nonconverged_trials=outcome.nonconverged_trials,
            best_macro_f1=outcome.best_macro_f1,
            best=dict(outcome.best),
            space=tuning.space,
            sampler=SAMPLER,
            pruner=PRUNER,
            seconds=seconds,
            train_notes=report.train_notes,
            train_patients=report.train_patients,
            diagnostic_strip=strip,
            text_preprocessing=row.input.text_preprocessing,
            text_normalization=normalization,
            environment=Environment(
                packages=installed_versions(PACKAGES), source=source
            ),
            hardware=hardware,
        ),
    )
    logger.info(
        "Ladění %s, fold %d: dokončených %d, prořezaných %d, čas %.3f s.",
        row.id,
        tuning.fold,
        outcome.completed_trials,
        outcome.pruned_trials,
        seconds,
    )
    return ExitCode.OK


def _validation_scorer(
    model: ModelName, normalization: TextNormalization | None, seed: Seed
) -> ValidationScorer:
    """Vybere adaptér rodiny deklarované řádkem matice.

    Returns:
        Scorer lineárního SVM.

    Raises:
        InvalidInputError: Rodina zatím nemá adaptér ladění.
    """
    if model is ModelName.SVM:
        return SklearnLinearSvmScorer(normalization, seed)
    msg = f"Pro model {model} zatím není scorer ladění."
    raise InvalidInputError(msg)
