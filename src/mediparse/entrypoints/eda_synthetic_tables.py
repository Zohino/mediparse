"""Vstupní bod tabulek EDA z korpusu syntetických zpráv s kanárkem."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.domain.language import Language
from mediparse.domain.synthetic_bhc import bhc_rows
from mediparse.domain.synthetic_mimic import corpus_tables, input_notes
from mediparse.domain.synthetic_sentinel import section_tokens, sentinel_tables
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from mediparse.infrastructure.duckdb_cohort import MIN_CHARS
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.mimic_table import TableShape
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.section_headers import SectionLexicon

logger = logging.getLogger(__name__)


def main() -> ExitCode:
    """Konzolový skript ``mediparse-eda-synthetic-tables``.

    Returns:
        Návratový kód.
    """
    configure_logging()
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Zapíše tabulky manifestu a značku s počtem propouštěcích zpráv.

    Returns:
        OK po zápisu; REFUSED pro chybějící či neplatný vstup nebo nespárovaný korpus.
    """
    args = _parser().parse_args(argv)
    shapes = read_manifest(args.manifest)
    plans = PlansFile(args.plans).load()
    notes = input_notes(
        plans, CorpusDirectory(args.corpus).notes(Language.EN), Language.EN
    )
    lexicon = load_section_lexicon(args.lexicon)
    tables = eda_tables(
        plans,
        notes,
        load_diagnosis_codes(args.candidates),
        lexicon,
        shapes,
        prefix=args.prefix,
        text_chars=MIN_CHARS,
    )
    for table, rows in tables.items():
        write_string_table(args.out / f"{table}.parquet", shapes[table].columns, rows)
    write_marker(args.out, SyntheticMarker(prefix=args.prefix, rows=len(notes)))
    logger.info("%d syntetických tabulek v %s", len(shapes), args.out)
    return ExitCode.OK


# Vstupy kompozice zůstávají explicitní, bez konfiguračního typu jen pro předávání polí.
def eda_tables(  # ruff: ignore[too-many-arguments]
    plans: Sequence[NotePlan],
    notes: Sequence[InputNote],
    codes: DiagnosisCodes,
    lexicon: SectionLexicon,
    shapes: Mapping[str, TableShape],
    *,
    prefix: str,
    text_chars: int,
) -> dict[str, list[dict[str, str]]]:
    """Složí tabulky EDA s tokeny částí a targety BHC bez čtení či zápisu souborů.

    Returns:
        Nové tabulky s kanárkem a sloupci v pořadí manifestu.

    Raises:
        InvalidInputError: Plány, pokrytí, prefix, lexikon nebo manifest jsou neplatné.
    """
    corpus = corpus_tables(plans, notes, codes)
    corpus["discharge"] = section_tokens(corpus["discharge"], lexicon, prefix)
    # Target má nést token, aby kanárek zachytil i výpis targetu jediné zprávy.
    corpus["mimic-iv-bhc"] = bhc_rows(plans, corpus["discharge"], lexicon)
    # Úplná validace před prvním zápisem brání napůl zapsanému adresáři při odmítnutí.
    return sentinel_tables(corpus, shapes, prefix, text_chars)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-eda-synthetic-tables",
        description="Syntetické tabulky ve tvaru MIMIC pro render EDA.",
    )
    for name in ("plans", "corpus", "manifest", "candidates", "lexicon", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    return parser
