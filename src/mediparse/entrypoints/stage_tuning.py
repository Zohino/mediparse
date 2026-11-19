"""Ladění CASH stupně na vývojové části mimo Snakemake.

Doporučené cesty: results/screening/classical_trials.parquet,
resources/mimic/screening/classical_trial_predictions.parquet (řádková data,
mimo results/), resources/mimic/screening/classical_trials_checkpoint a
results/screening/classical_trials_manifest.json. Nad MIMIC spouští jen uživatel.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.stage_tuning import StageTuning
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import StageTuningManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_corpus import (
    ModelCorpusPaths,
    checkpoint_key,
    model_corpus,
)
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.optuna_search import (
    STAGE_PRUNER,
    OptunaStageSearch,
    stage_sampler,
)
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.sklearn_stage import SklearnStageScorer
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.stage_config import load_stage_config
from mediparse.infrastructure.stage_trials_table import ParquetStageCheckpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.application.ports.experiment import Clock
    from mediparse.domain.environment import SourceRevision

logger = logging.getLogger(__name__)
PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "optuna",
    "pyarrow",
    "pydantic",
)


@dataclass(frozen=True, kw_only=True)
class StageTuningPaths:
    """Explicitní cesty vstupů, výstupů a checkpointu stupně."""

    corpus: ModelCorpusPaths
    config: Path
    out: Path
    predictions: Path
    checkpoint: Path
    manifest: Path


def main() -> ExitCode:
    """Načte povinné cesty a nastaví logování.

    Returns:
        Návratový kód ladění.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cohort",
        "split",
        "config",
        "candidates",
        "lexicon",
        "out",
        "predictions",
        "checkpoint",
        "manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    return run(
        environ=os.environ,
        paths=StageTuningPaths(
            corpus=ModelCorpusPaths(
                cohort=args.cohort,
                split=args.split,
                candidates=args.candidates,
                lexicon=args.lexicon,
            ),
            config=args.config,
            out=args.out,
            predictions=args.predictions,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
        ),
        revision=revision_source(os.environ, Path.cwd()),
        clock=PerfCounterClock(),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    paths: StageTuningPaths,
    revision: Callable[[], SourceRevision],
    clock: Clock,
) -> ExitCode:
    """Složí adaptéry a zapíše oba výstupy s manifestem.

    Returns:
        OK po dokončení, REFUSED při neplatných vstupech či nepovoleném přístupu.
    """
    check_inputs_access(environ, paths.corpus.data)
    config = load_stage_config(paths.config)
    packages = installed_versions(PACKAGES)
    source = revision()
    corpus = model_corpus(paths.corpus, config.input)
    hardware = machine_hardware(())
    report = StageTuning(
        table=corpus.table,
        split=corpus.split,
        partitioner=StratifiedGroupFolds(),
        search=OptunaStageSearch.create(config),
        scorer=SklearnStageScorer(
            config,
            corpus.normalization,
            config.lsa_components,
            config.seed,
            # Scratch používá jen fastText, který tento stupeň nepřijímá.
            paths.checkpoint,
        ),
        progress=ParquetStageCheckpoint(
            paths.checkpoint,
            checkpoint_key(paths.config, paths.corpus, corpus.dataset_sha256, packages),
            paths.out,
            paths.predictions,
        ),
        clock=clock,
    ).run(config)
    write_model(
        paths.manifest,
        StageTuningManifest(
            dataset_sha256=corpus.dataset_sha256,
            config_sha256=file_sha256(paths.config),
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            seed=config.seed,
            trials=config.trials,
            random_startup_trials=config.random_startup_trials,
            completed_trials=report.completed_trials,
            nonconverged_trials=report.nonconverged_trials,
            best_trial=report.best.number,
            best_branch=config.branch(report.best.params).name,
            best_params=dict(report.best.params),
            best_macro_f1=report.best.value,
            sampler=stage_sampler(config.startup_trials),
            pruner=STAGE_PRUNER,
            seconds=report.seconds,
            sample_notes=report.sample_notes,
            sample_patients=report.sample_patients,
            development_notes=report.development_notes,
            development_patients=report.development_patients,
            results_sha256=file_sha256(paths.out),
            predictions_sha256=file_sha256(paths.predictions),
            diagnostic_strip=corpus.strip,
            text_preprocessing=config.input.text_preprocessing,
            text_normalization=corpus.normalization,
            environment=Environment(packages=packages, source=source),
            hardware=hardware,
        ),
    )
    logger.info(
        "Ladění stupně: %d pokusů, vítěz %d (%s), čas %.3f s.",
        report.completed_trials,
        report.best.number,
        config.branch(report.best.params).name,
        report.seconds,
    )
    return ExitCode.OK
