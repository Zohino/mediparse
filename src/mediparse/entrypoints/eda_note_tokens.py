"""Připraví cache délek tokenů, přesného dělení a syntetických párů pro sešit 4."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.note_tokens import MeasureNoteTokens
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note_tokens import TranslationPrompt
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.eda_cache import is_current, pending_cache
from mediparse.infrastructure.eda_duckdb import EdaDatabase
from mediparse.infrastructure.hf_tokenizers import (
    counter_for,
    load_tokenizer,
    prompt_overhead,
    special_tokens,
    tokenizer_info,
)
from mediparse.infrastructure.note_texts_parquet import ParquetNoteTexts
from mediparse.infrastructure.note_tokens_input import (
    SAMPLE_LIMIT,
    SampleNoteTexts,
    read_parallel,
    tokens_key,
)
from mediparse.infrastructure.note_tokens_table import ARTIFACTS, ParquetNoteTokens
from mediparse.infrastructure.translation_script import ScriptSplitter, load_translation

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


# Explicitní cesty jsou veřejným kontraktem nástroje podle briefu.
def run(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    parquet: Path,
    translation: Path,
    english: Path,
    czech: Path,
    out: Path,
    environ: Mapping[str, str],
) -> bool:
    """Stráží vstup před čtením a hashováním, pak spočítá nebo použije cache.

    Returns:
        True po výpočtu; False při platné cache.

    Raises:
        DataAccessError: Prostředí nesmí číst skutečná data.
        InvalidInputError: Vstup nebo cache neodpovídá kontraktu.
    """
    database = EdaDatabase(parquet, environ)
    views = database.guarded_views()
    for table in ("discharge", "radiology"):
        if table not in views:
            msg = f"V adresáři tabulek chybí {table}.parquet."
            raise InvalidInputError(msg)
    script = load_translation(translation)
    infos = tokenizer_info(script)
    key = tokens_key(views, translation, english, czech, infos)
    directory = out / "tokens"
    artifacts = tuple(name for name, _ in ARTIFACTS)
    if is_current(directory, key, artifacts):
        logger.info("cache tokenů platí")
        return False
    pairs = read_parallel(english, czech)
    loaded = {
        info.tokenizer: load_tokenizer(info.model, info.revision) for info in infos
    }
    infos = [
        replace(info, special_tokens=special_tokens(loaded[info.tokenizer]))
        if info.role == "encoder"
        else info
        for info in infos
    ]
    prompt = TranslationPrompt(
        prompt_overhead(script, loaded["translategemma"]),
        script.prepare.SPLIT_RATIO,
        SAMPLE_LIMIT,
    )
    counters = {
        info.tokenizer: counter_for(info, loaded[info.tokenizer], parallel=False)
        for info in infos
    }
    parallel_counters = {
        info.tokenizer: counter_for(info, loaded[info.tokenizer], parallel=True)
        for info in infos
    }
    with (
        pending_cache(directory, key, artifacts) as temporary,
        ParquetNoteTokens(temporary).writer() as sink,
    ):
        sink.settings(infos, prompt)
        total = MeasureNoteTokens(
            {
                name: ParquetNoteTexts(views[name])
                for name in ("discharge", "radiology")
            },
            {
                name: SampleNoteTexts(database, name)
                for name in ("discharge", "radiology")
            },
            counters,
            parallel_counters,
            ScriptSplitter(script, loaded["translategemma"]),
            sink,
        ).run(pairs)
    logger.info("Délky tokenů zapsány: %d zpráv.", total)
    return True


def main() -> ExitCode:
    """Nastaví logování a načte explicitní cesty nástroje.

    Returns:
        OK při výpočtu i platné cache, REFUSED při odmítnutém vstupu.
    """
    configure_logging()
    return _run_cli()


@refusing_invalid_input
def _run_cli() -> ExitCode:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parquet", "translation", "english", "czech", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.parquet, args.translation, args.english, args.czech, args.out, os.environ)
    return ExitCode.OK
