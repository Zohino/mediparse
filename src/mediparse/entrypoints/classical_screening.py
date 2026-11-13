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
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.classical_screening import ClassicalScreening
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import ScreeningManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_input_table import (
    model_input_table,
    text_normalization,
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
from mediparse.infrastructure.split_table import ParquetSplitTable

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
        cohort=args.cohort,
        split=args.split,
        config=args.config,
        candidates=args.candidates,
        lexicon=args.lexicon,
        out=args.out,
        predictions=args.predictions,
        manifest=args.manifest,
        revision=revision_source(os.environ, Path.cwd()),
    )


@refusing_invalid_input
def run(  # ruff: ignore[too-many-arguments] - composition root předává explicitní cesty a zdroj revize
    *,
    environ: Mapping[str, str],
    cohort: Path,
    split: Path,
    config: Path,
    candidates: Path,
    lexicon: Path,
    out: Path,
    predictions: Path,
    manifest: Path,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    """Složí screening a zapíše agregáty i řádkové predikce s provenance.

    Returns:
        OK po dokončení mřížky; REFUSED při neplatných vstupech, nečitelné
        revizi nebo neoznačených datech v prostředí agenta.
    """
    check_inputs_access(environ, (cohort, split))
    screening = load_classical_screening_config(config)
    source = revision()
    table, strip = model_input_table(
        cohort=cohort, model_input=screening.input, lexicon=lexicon
    )
    normalization = text_normalization(candidates, screening.input)
    report = ClassicalScreening(
        table=table,
        split=ParquetSplitTable(split),
        partitioner=StratifiedGroupFolds(),
        featurizer=TfidfFeaturizer(
            normalization,
            screening_lsa(screening.families, screening.lsa_components, screening.seed),
        ),
        tuner=SklearnThresholdTuner(),
        scorer=SklearnScorer(),
        results=ParquetScreeningTable(out),
        predictions=ParquetScreeningPredictions(predictions),
    ).run(screening)
    write_model(
        manifest,
        ScreeningManifest(
            dataset_sha256=file_sha256(cohort),
            split_seed=report.split_config.seed,
            split_folds=report.split_config.folds,
            split_sha256=report.split_sha256,
            config_sha256=file_sha256(config),
            seed=screening.seed,
            results_sha256=file_sha256(out),
            predictions_sha256=file_sha256(predictions),
            development_notes=report.development_notes,
            development_patients=report.development_patients,
            diagnostic_strip=strip,
            text_preprocessing=screening.input.text_preprocessing,
            text_normalization=normalization,
            environment=Environment(
                packages=installed_versions(PACKAGES), source=source
            ),
        ),
    )
    logger.info(
        "Screening: %d vývojových zpráv a %d pacientů, výsledky zapsány do %s.",
        report.development_notes,
        report.development_patients,
        out,
    )
    return ExitCode.OK
