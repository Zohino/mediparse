"""Ladění CASH stupně na vývojové části mimo Snakemake.

Pro stupně classical, embeddings a networks jsou doporučené cesty:
results/screening/<stupeň>_trials.parquet,
resources/mimic/screening/<stupeň>_trial_predictions.parquet (řádková data),
resources/mimic/screening/<stupeň>_trials_checkpoint a
results/screening/<stupeň>_trials_manifest.json.
Cache: --embeddings resources/mimic/screening/embeddings.
Dočasné tokeny: --scratch resources/mimic/screening/scratch.
Nad MIMIC spouští jen uživatel; řádková data patří mimo results/.
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
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import StageTuningManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_corpus import (
    ModelCorpusPaths,
    checkpoint_key,
    model_corpus,
    text_normalization,
)
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.hf_encoder import (
    ENCODER_PACKAGES,
    HfMeanEncoder,
    load_encoder,
)
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.note_embeddings import NoteEmbeddings
from mediparse.infrastructure.nvml_gpu import gpu_monitor
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
    from mediparse.domain.representation import EncoderSpec

logger = logging.getLogger(__name__)
PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "fasttext",
    "gensim",
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
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Cache embeddingů; doporučeno resources/mimic/screening/embeddings, mimo results/.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        required=True,
        help="Dočasné tokeny zpráv; doporučeno resources/mimic/screening/scratch, mimo checkpoint a results/.",
    )
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
            embeddings=args.embeddings,
            scratch=args.scratch,
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
    encoders: Callable[[EncoderSpec], HfMeanEncoder] = load_encoder,
) -> ExitCode:
    """Složí adaptéry a zapíše oba výstupy s manifestem.

    Returns:
        OK po dokončení, REFUSED při neplatných vstupech či nepovoleném přístupu.
    """
    check_inputs_access(environ, paths.corpus.data)
    _check_work_paths(paths)
    config = load_stage_config(paths.config)
    versions = installed_versions(ENCODER_PACKAGES) if config.encoders else {}
    packages = installed_versions(PACKAGES) | versions
    source = revision()
    corpus = model_corpus(
        paths.corpus, strip_diagnostic_sections=config.input.strip_diagnostic_sections
    )
    normalization = text_normalization(
        paths.corpus.candidates, config.input.text_preprocessing
    )
    with gpu_monitor(
        environ.get("CUDA_VISIBLE_DEVICES"), environ.get("CUDA_DEVICE_ORDER")
    ) as monitor:
        hardware = machine_hardware(monitor.devices)
        report = StageTuning(
            table=corpus.table,
            split=corpus.split,
            partitioner=StratifiedGroupFolds(),
            search=OptunaStageSearch.create(config),
            scorer=SklearnStageScorer(
                config,
                normalization,
                config.seed,
                paths.scratch,
                NoteEmbeddings(
                    paths.embeddings, versions, lambda spec: encoders(spec).embed
                ),
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
            text_normalization=normalization,
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


def _check_work_paths(paths: StageTuningPaths) -> None:
    # Pracovní adaptéry zakládají adresáře dřív než checkpoint zapíše svůj klíč.
    for path in (paths.scratch, paths.embeddings):
        if path.resolve().is_relative_to(paths.checkpoint.resolve()):
            msg = "Scratch a cache embeddingů musí ležet mimo adresář checkpointu."
            raise InvalidInputError(msg)
