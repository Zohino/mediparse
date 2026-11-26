"""Kontrola kandidátů a doladění stupně na celé vývojové části.

Doporučené cesty: results/refinement/<stupeň>_trials.parquet,
resources/mimic/refinement/<stupeň>_trial_predictions.parquet (řádková data),
resources/mimic/refinement/<stupeň>_checkpoint a results/refinement/<stupeň>_manifest.json.
Spuštění: pixi run --frozen -e gpu mediparse-stage-refinement …
Nad MIMIC spouští jen uživatel.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.stage_refinement import StageRefinement
from mediparse.domain.environment import Environment
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import (
    RefinementCandidate,
    StageRefinementManifest,
    ensure_stage_lineage,
)
from mediparse.domain.stage_refinement import widen
from mediparse.domain.stage_tuning import BRANCH_KEY, family_candidates
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
from mediparse.infrastructure.matrix_manifest import load_matrix
from mediparse.infrastructure.nvml_gpu import gpu_monitor
from mediparse.infrastructure.optuna_search import (
    STAGE_PRUNER,
    OptunaStageSearch,
    stage_sampler,
)
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.run_manifest_file import load_stage_tuning_manifest
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.stage_config import (
    load_refinement_config,
    load_stage_config,
)
from mediparse.infrastructure.stage_trials_table import (
    ParquetStageCheckpoint,
    read_trial_records,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.application.ports.experiment import Clock
    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.representation import EncoderSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class StageRefinementPaths:
    """Explicitní cesty vstupů, výstupů a checkpointu stupně."""

    corpus: ModelCorpusPaths
    config: Path
    stage_results: Path
    stage_manifest: Path
    refinement: Path
    matrix: Path
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
    for name in (
        "stage-results",
        "stage-manifest",
        "refinement",
        "matrix",
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
    add_stage_work_arguments(parser)
    args = parser.parse_args()
    return run(
        environ=os.environ,
        paths=StageRefinementPaths(
            corpus=ModelCorpusPaths(
                cohort=args.cohort,
                split=args.split,
                candidates=args.candidates,
                lexicon=args.lexicon,
            ),
            stage_results=args.stage_results,
            stage_manifest=args.stage_manifest,
            refinement=args.refinement,
            matrix=args.matrix,
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


# Explicitní vstupy oddělují cesty, zdroje a provozní paralelitu.
@refusing_invalid_input
def run(  # ruff: ignore[too-many-arguments]
    *,
    environ: Mapping[str, str],
    paths: StageRefinementPaths,
    revision: Callable[[], SourceRevision],
    clock: Clock,
    encoders: Callable[[EncoderSpec], HfMeanEncoder] = load_encoder,
    jobs: int = 1,
) -> ExitCode:
    """Ověří předkolo, složí adaptéry a uloží vítěze stupně.

    Returns:
        OK po dokončení, REFUSED při neplatných vstupech či nepovoleném přístupu.

    Raises:
        InvalidInputError: Rozšíření prostoru není platné; dekorátor vrátí REFUSED.
    """
    check_inputs_access(environ, paths.corpus.data)
    check_work_paths(
        scratch=paths.scratch,
        embeddings=paths.embeddings,
        checkpoint=paths.checkpoint,
    )
    stage = load_stage_config(paths.config)
    refinement = load_refinement_config(paths.refinement)
    matrix = load_matrix(paths.matrix)
    screening = read_trial_records(paths.stage_results)
    candidates = family_candidates(screening)
    corpus = model_corpus(
        paths.corpus, strip_diagnostic_sections=stage.strip_diagnostic_sections
    )
    normalizations = stage_normalizations(paths.corpus.candidates, stage)
    ensure_stage_lineage(
        load_stage_tuning_manifest(paths.stage_manifest),
        config_sha256=file_sha256(paths.config),
        results_sha256=file_sha256(paths.stage_results),
        dataset_sha256=corpus.dataset_sha256,
        split_sha256=corpus.split.read().split.sha256(),
        diagnostic_strip=corpus.strip,
        text_normalization=normalizations,
        best_trial=candidates[0].number,
    )
    try:
        config = widen(stage, refinement)
    except ValueError as error:
        msg = f"Config doladění {paths.refinement}: {error}"
        raise InvalidInputError(msg) from error
    environment = Environment(packages=stage_packages(config), source=revision())
    with gpu_monitor(
        environ.get("CUDA_VISIBLE_DEVICES"), environ.get("CUDA_DEVICE_ORDER")
    ) as monitor:
        hardware = machine_hardware(monitor.devices)
        report = StageRefinement(
            table=corpus.table,
            split=corpus.split,
            partitioner=StratifiedGroupFolds(),
            searches=OptunaStageSearch,
            scorer=stage_scorer(
                config,
                normalizations,
                embeddings=paths.embeddings,
                scratch=paths.scratch,
                versions=environment.packages,
                encoders=encoders,
                jobs=jobs,
            ),
            progress=ParquetStageCheckpoint(
                paths.checkpoint,
                checkpoint_key(
                    paths.config,
                    paths.corpus,
                    corpus.dataset_sha256,
                    environment.packages,
                    extra=(
                        paths.refinement,
                        paths.matrix,
                        paths.stage_results,
                        paths.stage_manifest,
                    ),
                ),
                paths.out,
                paths.predictions,
            ),
            clock=clock,
        ).run(config, refinement, screening, matrix.tuning_trials)
    first, second = (
        RefinementCandidate(
            stage_trial=candidate.number,
            branch=str(candidate.params[BRANCH_KEY]),
            params=dict(candidate.params),
            stage_macro_f1=candidate.value,
            check_macro_f1=check.value,
        )
        for candidate, check in zip(candidates, report.checks, strict=True)
    )
    write_model(
        paths.manifest,
        StageRefinementManifest(
            jobs=jobs,
            encoders=config.encoders,
            dataset_sha256=corpus.dataset_sha256,
            stage_config_sha256=file_sha256(paths.config),
            refinement_config_sha256=file_sha256(paths.refinement),
            matrix_sha256=file_sha256(paths.matrix),
            stage_manifest_sha256=file_sha256(paths.stage_manifest),
            stage_results_sha256=file_sha256(paths.stage_results),
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            seed=config.seed,
            folds=stage.folds,
            tuning_trials=matrix.tuning_trials,
            random_startup_trials=refinement.random_startup_trials,
            candidates=(first, second),
            best_trial=report.best.number,
            best_params=dict(report.best.params),
            best_macro_f1=report.best.value,
            completed_trials=report.completed_trials,
            nonconverged_trials=report.nonconverged_trials,
            sampler=stage_sampler(report.study.startup_trials),
            pruner=STAGE_PRUNER,
            seconds=report.seconds,
            development_notes=report.development_notes,
            development_patients=report.development_patients,
            results_sha256=file_sha256(paths.out),
            predictions_sha256=file_sha256(paths.predictions),
            diagnostic_strip=corpus.strip,
            text_preprocessing=config.text_preprocessing,
            text_normalization=normalizations,
            environment=environment,
            hardware=hardware,
        ),
    )
    logger.info(
        "Doladění stupně: %d pokusů, vítěz %d (%s), čas %.3f s.",
        report.completed_trials,
        report.best.number,
        report.study.branches[0].name,
        report.seconds,
    )
    return ExitCode.OK
