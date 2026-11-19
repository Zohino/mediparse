"""Screening stupně na vývojové části mimo pipeline Snakemake.

Doporučené výstupy jsou ``results/screening/classical.parquet`` a
``results/screening/classical_manifest.json``. Nad MIMIC nástroj spouští jen uživatel.
Predikce patří do ``resources/mimic/screening/classical_predictions.parquet``:
jde o řádková data, která do ``results/`` nesmí.
Rozpracovaný běh patří do ``resources/mimic/screening/classical_checkpoint``.
Mělký stupeň používá obdobné cesty ``results/screening/shallow.parquet`` a
``resources/mimic/screening/shallow_checkpoint``.
Po pádu stačí spustit týž příkaz.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.screening import Screening
from mediparse.domain.environment import Environment
from mediparse.domain.run_manifest import ScreeningManifest
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_corpus import (
    ModelCorpusPaths,
    model_corpus,
)
from mediparse.infrastructure.code_fingerprint import code_key
from mediparse.infrastructure.data_access_guard import check_inputs_access
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.hf_encoder import (
    ENCODER_PACKAGES,
    HfMeanEncoder,
    load_encoder,
)
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.note_embeddings import NoteEmbeddings
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.screening_config import load_screening_config
from mediparse.infrastructure.screening_table import (
    ParquetScreeningCheckpoint,
)
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.sklearn_screening import (
    EncoderFeaturizer,
    ScreeningFeaturizer,
    SklearnThresholdTuner,
    TfidfFeaturizer,
    TokenFeaturizer,
    screening_lsa,
)
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.source_revision import revision_source

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.screening import EncoderSpec

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
    """Vstupní a výstupní cesty screeningu stupně."""

    corpus: ModelCorpusPaths
    config: Path
    out: Path
    predictions: Path
    manifest: Path
    checkpoint: Path
    embeddings: Path
    scratch: Path


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Rozpracovaný běh s řádkovými díly; doporučeno resources/mimic/screening/classical_checkpoint, mimo results/.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Cache embeddingů; doporučeno resources/mimic/screening/embeddings, mimo results/.",
    )
    parser.add_argument("--scratch", type=Path, required=True)
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
            checkpoint=args.checkpoint,
            embeddings=args.embeddings,
            scratch=args.scratch,
        ),
        revision=revision_source(os.environ, Path.cwd()),
    )


@refusing_invalid_input
def run(
    *,
    environ: Mapping[str, str],
    paths: ScreeningPaths,
    revision: Callable[[], SourceRevision],
    encoders: Callable[[EncoderSpec], HfMeanEncoder] = load_encoder,
) -> ExitCode:
    """Složí screening a zapíše agregáty i řádkové predikce s provenance.

    Returns:
        OK po dokončení mřížky; REFUSED při neplatných vstupech, nečitelné
        revizi nebo neoznačených datech v prostředí agenta.
    """
    check_inputs_access(environ, paths.corpus.data)
    screening = load_screening_config(paths.config)
    versions = installed_versions(ENCODER_PACKAGES) if screening.encoders else {}
    packages = installed_versions(PACKAGES) | versions
    source = revision()
    corpus = model_corpus(paths.corpus, screening.input)
    report = Screening(
        table=corpus.table,
        split=corpus.split,
        partitioner=StratifiedGroupFolds(),
        featurizer=ScreeningFeaturizer(
            TfidfFeaturizer(
                corpus.normalization,
                screening_lsa(
                    screening.families, screening.lsa_components, screening.seed
                ),
            ),
            EncoderFeaturizer(
                NoteEmbeddings(
                    paths.embeddings, versions, lambda spec: encoders(spec).embed
                )
            ),
            tokens=TokenFeaturizer(corpus.normalization),
        ),
        tuner=SklearnThresholdTuner(paths.scratch),
        scorer=SklearnScorer(),
        progress=ParquetScreeningCheckpoint(
            directory=paths.checkpoint,
            key=checkpoint_key(paths, corpus.dataset_sha256, packages),
            results=paths.out,
            predictions=paths.predictions,
        ),
    ).run(screening)
    write_model(
        paths.manifest,
        ScreeningManifest(
            encoders=screening.encoders,
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
            environment=Environment(packages=packages, source=source),
        ),
    )
    logger.info(
        "Screening: %d vývojových zpráv a %d pacientů, výsledky zapsány do %s.",
        report.development_notes,
        report.development_patients,
        paths.out,
    )
    return ExitCode.OK


def checkpoint_key(
    paths: ScreeningPaths,
    dataset_sha256: str,
    packages: Mapping[str, str],
) -> str:
    """Otiskne vstupy, kód a verze knihoven pro obnovu screeningu.

    Returns:
        Klíč rozpracovaného běhu.
    """
    return code_key([
        ("config", file_sha256(paths.config).encode()),
        ("cohort", dataset_sha256.encode()),
        ("split", file_sha256(paths.corpus.split).encode()),
        ("candidates", file_sha256(paths.corpus.candidates).encode()),
        ("lexicon", file_sha256(paths.corpus.lexicon).encode()),
        *((f"packages/{name}", value.encode()) for name, value in packages.items()),
    ])
