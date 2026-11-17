"""Klasický screening vývojové části mimo pipeline Snakemake.

Doporučené výstupy jsou ``results/screening/classical.parquet`` a
``results/screening/classical_manifest.json``. Nad MIMIC nástroj spouští jen uživatel.
Predikce patří do ``resources/mimic/screening/classical_predictions.parquet``:
jde o řádková data, která do ``results/`` nesmí.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.classical_screening import ClassicalScreening
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import ScreeningManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_input_table import (
    ModelCorpusPaths,
    model_corpus,
)
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.screening_config import load_classical_screening_config
from mediparse.infrastructure.screening_table import (
    ParquetScreeningPredictions,
    ParquetScreeningTable,
)
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.sklearn_screening import (
    SklearnThresholdTuner,
    TfidfFeaturizer,
    screening_lsa,
)
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.source_revision import revision_source

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.domain.environment import SourceRevision

logger = logging.getLogger(__name__)
PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "pyarrow",
    "pydantic",
)


@dataclass(frozen=True, kw_only=True)
class ScreeningPaths:
    """Vstupní a výstupní cesty screeningu klasických modelů."""

    corpus: ModelCorpusPaths
    config: Path
    out: Path
    predictions: Path
    manifest: Path


def main() -> ExitCode:
    """Načte explicitní cesty screeningu a nastaví logování.

    Returns:
        OK po zápisu výsledků, predikcí a manifestu, REFUSED při odmítnutí vstupů.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=(
            "Řádkové predikce; doporučeno "
            "resources/mimic/screening/classical_predictions.parquet, mimo results/."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    return run(
        environ=os.environ,
        paths=ScreeningPaths(
            corpus=ModelCorpusPaths(
                cohort=args.cohort,
                candidates=args.candidates,
                lexicon=args.lexicon,
                split=args.split,
            ),
            config=args.config,
            out=args.out,
            predictions=args.predictions,
            manifest=args.manifest,
        ),
        revision=revision_source(os.environ, Path.cwd()),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    paths: ScreeningPaths,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    """Složí screening a zapíše agregáty i řádkové predikce s provenance.

    Returns:
        OK po dokončení mřížky; REFUSED při neplatných vstupech, nečitelné
        revizi nebo neoznačených datech v prostředí agenta.
    """
    check_inputs_access(environ, paths.corpus.data)
    screening = load_classical_screening_config(paths.config)
    source = revision()
    corpus = model_corpus(paths.corpus, screening.input)
    report = ClassicalScreening(
        table=corpus.table,
        split=corpus.split,
        partitioner=StratifiedGroupFolds(),
        featurizer=TfidfFeaturizer(
            corpus.normalization,
            screening_lsa(screening.families, screening.lsa_components, screening.seed),
        ),
        tuner=SklearnThresholdTuner(),
        scorer=SklearnScorer(),
        results=ParquetScreeningTable(paths.out),
        predictions=ParquetScreeningPredictions(paths.predictions),
    ).run(screening)
    write_model(
        paths.manifest,
        ScreeningManifest(
            dataset_sha256=corpus.dataset_sha256,
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            config_sha256=file_sha256(paths.config),
            seed=screening.seed,
            results_sha256=file_sha256(paths.out),
            predictions_sha256=file_sha256(paths.predictions),
            development_notes=report.development_notes,
            development_patients=report.development_patients,
            diagnostic_strip=corpus.strip,
            text_preprocessing=screening.input.text_preprocessing,
            text_normalization=corpus.normalization,
            environment=Environment(
                packages=installed_versions(PACKAGES), source=source
            ),
        ),
    )
    logger.info(
        "Screening: %d vývojových zpráv a %d pacientů, výsledky zapsány do %s.",
        report.development_notes,
        report.development_patients,
        paths.out,
    )
    return ExitCode.OK
