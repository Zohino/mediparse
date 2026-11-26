"""Ladění CASH stupně na vývojové části mimo Snakemake.

Pro stupně classical, embeddings a networks jsou doporučené cesty:
results/screening/<stupeň>_trials.parquet,
resources/mimic/screening/<stupeň>_trial_predictions.parquet (řádková data),
resources/mimic/screening/<stupeň>_trials_checkpoint a
results/screening/<stupeň>_trials_manifest.json.
Cache: --embeddings resources/mimic/screening/embeddings.
Dočasné tokeny zpráv a memmapy matic paralelních fitů: --scratch resources/mimic/screening/scratch.
Stupeň se sítí: pixi run --frozen -e gpu mediparse-stage-tuning …
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
from mediparse.domain.diagnosis import Diagnosis
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
from mediparse.infrastructure.torch_device import NETWORK_PACKAGES, require_device

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.application.ports.experiment import Clock
    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.representation import EncoderSpec
    from mediparse.domain.stage_tuning import StageTuningConfig

logger = logging.getLogger(__name__)
PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "fasttext",
    "gensim",
    "joblib",
    "threadpoolctl",
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
        help="Dočasné tokeny zpráv a memmapy matic paralelních fitů; doporučeno resources/mimic/screening/scratch, mimo checkpoint a results/.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        choices=range(1, len(Diagnosis) + 1),
        default=3,
        help=(
            "Procesy nezávislých fitů pokusu: diagnózy u klasiky a fastTextu, "
            "foldy u MultiResCNN (nejvýš jeden fold na GPU, fold i na GPU i mod počet GPU). "
            "Výsledky na počtu nezávisí."
        ),
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
        scratch=paths.scratch, embeddings=paths.embeddings, checkpoint=paths.checkpoint
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


def check_work_paths(*, scratch: Path, embeddings: Path, checkpoint: Path) -> None:
    """Ověří oddělení dočasných souborů od checkpointu.

    Raises:
        InvalidInputError: Scratch nebo cache leží uvnitř checkpointu.
    """
    # Pracovní adaptéry zakládají adresáře dřív než checkpoint zapíše svůj klíč.
    for path in (scratch, embeddings):
        if path.resolve().is_relative_to(checkpoint.resolve()):
            msg = "Scratch a cache embeddingů musí ležet mimo adresář checkpointu."
            raise InvalidInputError(msg)


def stage_normalizations(
    candidates: Path, config: StageTuningConfig
) -> dict[str, TextNormalization | None]:
    """Načte slovníky reprezentací společné pro scorer i manifest.

    Returns:
        Normalizace podle jména reprezentace.
    """
    return {
        name: text_normalization(candidates, level)
        for name, level in config.text_preprocessing.items()
    }


def stage_packages(config: StageTuningConfig) -> dict[str, str]:
    """Ověří zařízení a zjistí verze knihoven stupně.

    Returns:
        Balíčky pro manifest a klíč checkpointu.
    """
    for spec in config.encoders:
        require_device(spec.device)
    if config.network is not None:
        require_device(config.network.device)
    versions = installed_versions(ENCODER_PACKAGES) if config.encoders else {}
    packages = installed_versions(PACKAGES) | versions
    if config.network is not None:
        packages |= installed_versions(NETWORK_PACKAGES)
    return packages


# Cesty, encoder a počet procesů jsou nezávislé provozní vstupy composition root.
def stage_scorer(  # ruff: ignore[too-many-arguments]
    config: StageTuningConfig,
    normalizations: Mapping[str, TextNormalization | None],
    *,
    embeddings: Path,
    scratch: Path,
    versions: Mapping[str, str],
    encoders: Callable[[EncoderSpec], HfMeanEncoder],
    jobs: int,
) -> SklearnStageScorer:
    """Složí scorer se společnou cache embeddingů.

    Returns:
        Adaptér pro kontrolu i ladění bodů stupně.
    """
    return SklearnStageScorer(
        config,
        normalizations,
        config.seed,
        scratch,
        NoteEmbeddings(
            embeddings,
            {name: versions[name] for name in ENCODER_PACKAGES}
            if config.encoders
            else {},
            lambda spec: encoders(spec).embed,
        ),
        jobs=jobs,
    )
