"""Ladění CASH stupně na vývojové části mimo Snakemake.

Pro stupně classical, embeddings, networks a transformers jsou doporučené cesty:
results/screening/<stupeň>_trials.parquet,
resources/mimic/screening/<stupeň>_trial_predictions.parquet (řádková data),
resources/mimic/screening/<stupeň>_trials_checkpoint a
results/screening/<stupeň>_trials_manifest.json.
Cache: --embeddings resources/mimic/screening/embeddings.
Dočasné tokeny zpráv a memmapy matic paralelních fitů: --scratch resources/mimic/screening/scratch.
Stupeň se sítí nebo transformerem: pixi run --frozen -e gpu mediparse-stage-tuning …
Nad MIMIC spouští jen uživatel; řádková data patří mimo results/.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from mediparse.entrypoints.stage_composition import (
    add_stage_work_arguments,
    check_work_paths,
    stage_normalizations,
    stage_packages,
    stage_scorer,
)
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.hf_encoder import (
    HfMeanEncoder,
    load_encoder,
)
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.nvml_gpu import gpu_monitor
from mediparse.infrastructure.optuna_search import (
    STAGE_PRUNER,
    OptunaStageSearch,
    stage_sampler,
)
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.stage_config import load_stage_config
from mediparse.infrastructure.stage_trials_table import ParquetStageCheckpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.application.ports.experiment import Clock
    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.representation import EncoderSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class StageTuningPaths:
    """Explicitní cesty vstupů, výstupů a checkpointu stupně."""

    corpus: ModelCorpusPaths
    config: Path
    out: Path
    predictions: Path
    checkpoint: Path
    manifest: Path
    embeddings: Path
    scratch: Path


def main() -> ExitCode:
    """Načte povinné cesty a nastaví logování.

    Returns:
        Návratový kód ladění.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    ModelCorpusPaths.add_arguments(parser)
    for name in (
        "config",
        "out",
        "predictions",
        "checkpoint",
        "manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    add_stage_work_arguments(parser)
    args = parser.parse_args()
    return run(
        environ=os.environ,
        paths=StageTuningPaths(
            corpus=ModelCorpusPaths.from_arguments(args),
            config=args.config,
            out=args.out,
            predictions=args.predictions,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            embeddings=args.embeddings,
            scratch=args.scratch,
        ),
        revision=revision_source(os.environ, Path.cwd()),
        clock=PerfCounterClock(),
        jobs=args.jobs,
    )


# Explicitní vstupy composition root oddělují cesty, zdroje a provozní paralelitu.
@refusing_invalid_input
def run(  # ruff: ignore[too-many-arguments]
    *,
    environ: Mapping[str, str],
    paths: StageTuningPaths,
    revision: Callable[[], SourceRevision],
    clock: Clock,
    encoders: Callable[[EncoderSpec], HfMeanEncoder] = load_encoder,
    jobs: int = 1,
) -> ExitCode:
    """Složí adaptéry a zapíše oba výstupy s manifestem.

    Returns:
        OK po dokončení, REFUSED při neplatných vstupech či nepovoleném přístupu.
    """
    check_inputs_access(environ, paths.corpus.data)
    check_work_paths(
        scratch=paths.scratch,
        embeddings=paths.embeddings,
        checkpoint=paths.checkpoint,
    )
    config = load_stage_config(paths.config)
    packages = stage_packages(config)
    source = revision()
    corpus = model_corpus(
        paths.corpus, strip_diagnostic_sections=config.strip_diagnostic_sections
    )
    normalizations = stage_normalizations(paths.corpus.candidates, config)
    with gpu_monitor(
        environ.get("CUDA_VISIBLE_DEVICES"), environ.get("CUDA_DEVICE_ORDER")
    ) as monitor:
        hardware = machine_hardware(monitor.devices)
        report = StageTuning(
            table=corpus.table,
            split=corpus.split,
            partitioner=StratifiedGroupFolds(),
            search=OptunaStageSearch.create(config),
            scorer=stage_scorer(
                config,
                normalizations,
                embeddings=paths.embeddings,
                scratch=paths.scratch,
                versions=packages,
                encoders=encoders,
                jobs=jobs,
            ),
            progress=ParquetStageCheckpoint(
                paths.checkpoint,
                checkpoint_key(
                    paths.config, paths.corpus, corpus.dataset_sha256, packages
                ),
                paths.out,
                paths.predictions,
            ),
            clock=clock,
        ).run(config)
    write_model(
        paths.manifest,
        StageTuningManifest(
            jobs=jobs,
            encoders=config.encoders,
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
            sampler=stage_sampler(config.sampler_startup_trials),
            pruner=STAGE_PRUNER,
            seconds=report.seconds,
            sample_notes=report.sample_notes,
            sample_patients=report.sample_patients,
            development_notes=report.development_notes,
            development_patients=report.development_patients,
            results_sha256=file_sha256(paths.out),
            predictions_sha256=file_sha256(paths.predictions),
            diagnostic_strip=corpus.strip,
            text_preprocessing=config.text_preprocessing,
            text_normalization=normalizations,
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
