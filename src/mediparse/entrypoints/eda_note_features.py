"""Nástroj pro přípravu cache měření zpráv před renderem EDA."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.note_features import ComputeNoteFeatures
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.diagnosis_candidates_file import (
    load_diagnosis_variants,
    load_text_normalization,
)
from mediparse.infrastructure.eda_cache import (
    is_current,
    pending_cache,
)
from mediparse.infrastructure.eda_duckdb import EdaDatabase
from mediparse.infrastructure.note_features_table import (
    ARTIFACTS,
    ParquetNoteFeatures,
    note_features_key,
)
from mediparse.infrastructure.note_texts_parquet import ParquetNoteTexts
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def run(
    parquet: Path,
    lexicon: Path,
    candidates: Path,
    out: Path,
    environ: Mapping[str, str],
) -> bool:
    """Složí strážený výpočet, nebo použije cache se stejným klíčem.

    Returns:
        True, pokud se měření počítalo; False při platné cache.

    Raises:
        InvalidInputError: Vstup nebo cache neodpovídá kontraktu.
        DataAccessError: Prostředí nesmí číst skutečná data.
    """
    views = EdaDatabase(parquet, environ).guarded_views()
    if "discharge" not in views:
        msg = "V adresáři tabulek chybí discharge.parquet."
        raise InvalidInputError(msg)
    discharge = views["discharge"]
    key = note_features_key(discharge, lexicon, candidates)
    artifacts = tuple(name for name, _ in ARTIFACTS)
    if is_current(out, key, artifacts):
        logger.info("cache platí")
        return False
    parsed = load_section_lexicon(lexicon)
    normalization = load_text_normalization(candidates, variants=True)
    variant_map = load_diagnosis_variants(candidates)
    with (
        pending_cache(out, key, artifacts) as temporary,
        ParquetNoteFeatures(temporary).writer() as sink,
    ):
        total = ComputeNoteFeatures(
            ParquetNoteTexts(discharge),
            sink,
            tfidf_vectorizer(None).build_analyzer(),
            tfidf_vectorizer(normalization).build_analyzer(),
        ).run(parsed, normalization, variant_map)
    logger.info("Měření zpráv zapsáno: %d zpráv.", total)
    return True


def main() -> ExitCode:
    """Načte explicitní cesty nástroje a nastaví logování.

    Returns:
        OK při výpočtu i platné cache; REFUSED při odmítnutí vstupu.
    """
    configure_logging()
    return _run_cli()


@refusing_invalid_input
def _run_cli() -> ExitCode:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.parquet, args.lexicon, args.candidates, args.out, os.environ)
    return ExitCode.OK
