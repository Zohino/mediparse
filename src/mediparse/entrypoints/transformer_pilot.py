"""Pilot transformeru mimo Snakemake.

Doporučené výstupy: results/pilot/transformer_pilot.parquet a
results/pilot/transformer_pilot_manifest.json. Tabulka obsahuje jen agregáty.
Nad MIMIC spouští jen uživatel; pixi run --frozen -e gpu mediparse-transformer-pilot …
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.transformer_pilot import TransformerPilot
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import TransformerPilotManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_corpus import ModelCorpusPaths, model_corpus
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.hf_encoder import ENCODER_PACKAGES
from mediparse.infrastructure.hf_pilot_trainer import HfPilotTrainer
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.nvml_gpu import gpu_monitor
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.perf_clock import PerfCounterClock
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.stage_config import load_pilot_config
from mediparse.infrastructure.torch_device import require_device
from mediparse.infrastructure.transformer_pilot_table import ParquetPilotSink

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.application.ports.experiment import Clock, PilotTrainer
    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.seed import Seed


@dataclass(frozen=True, kw_only=True)
class TransformerPilotPaths:
    """Explicitní vstupy a veřejné výstupy pilotu."""

    corpus: ModelCorpusPaths
    config: Path
    out: Path
    manifest: Path


def main() -> ExitCode:
    """Načte cesty a nastaví logování.

    Returns:
        Návratový kód pilotu.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cohort",
        "split",
        "candidates",
        "lexicon",
        "config",
        "out",
        "manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    return run(
        environ=os.environ,
        paths=TransformerPilotPaths(
            corpus=ModelCorpusPaths(
                cohort=args.cohort,
                split=args.split,
                candidates=args.candidates,
                lexicon=args.lexicon,
            ),
            config=args.config,
            out=args.out,
            manifest=args.manifest,
        ),
        revision=revision_source(os.environ, Path.cwd()),
        clock=PerfCounterClock(),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    paths: TransformerPilotPaths,
    revision: Callable[[], SourceRevision],
    clock: Clock,
    trainer_factory: Callable[[Seed], PilotTrainer] = HfPilotTrainer,
) -> ExitCode:
    """Složí pilot po kontrole přístupu a dostupnosti encoderů.

    Returns:
        OK po zápisu agregátů a manifestu, jinak REFUSED.
    """
    check_inputs_access(environ, paths.corpus.data)
    config = load_pilot_config(paths.config)
    for variant in config.runs:
        require_device(variant.encoder.device)
    packages = installed_versions((
        "python",
        "numpy",
        "scikit-learn",
        "pyarrow",
        "pydantic",
        *ENCODER_PACKAGES,
    ))
    source = revision()
    corpus = model_corpus(
        paths.corpus, strip_diagnostic_sections=config.strip_diagnostic_sections
    )
    with gpu_monitor(
        environ.get("CUDA_VISIBLE_DEVICES"), environ.get("CUDA_DEVICE_ORDER")
    ) as monitor:
        hardware = machine_hardware(monitor.devices)
        report = TransformerPilot(
            table=corpus.table,
            split=corpus.split,
            partitioner=StratifiedGroupFolds(),
            trainer=trainer_factory(config.seed),
            sink=ParquetPilotSink(paths.out),
            clock=clock,
        ).run(config)
    write_model(
        paths.manifest,
        TransformerPilotManifest(
            encoders=tuple(variant.encoder for variant in config.runs),
            dataset_sha256=corpus.dataset_sha256,
            config_sha256=file_sha256(paths.config),
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            seed=config.seed,
            seconds=report.seconds,
            sample_notes=report.sample_notes,
            sample_patients=report.sample_patients,
            development_notes=report.development_notes,
            development_patients=report.development_patients,
            results_sha256=file_sha256(paths.out),
            diagnostic_strip=corpus.strip,
            environment=Environment(packages=packages, source=source),
            hardware=hardware,
        ),
    )
    return ExitCode.OK
