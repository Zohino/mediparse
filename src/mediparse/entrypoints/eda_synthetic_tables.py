"""Vstupní bod syntetických tabulek pro render EDA: tabulky ve tvaru MIMIC bez dat MIMIC.

Konzolový skript ``mediparse-eda-synthetic-tables`` čte veřejný manifest validace, config
kandidátů (předpony kódů, klíčová slova a zkratky pětice), lexikon sekcí a config zmínek
(negace a příznaky) a zapíše pro každou tabulku manifestu
parquet se samými řetězci. Mimo typové sloupce je každá buňka v tabulce jedinečná; typové
sloupce nesou slovník MIMIC po blocích. Klíče pacienta a hospitalizace sdílí tabulky jako
v MIMIC, propouštěcí zprávy mají preambuli a sekce lexikonu a projdou filtrem délky kohorty.
Render dokumentů tak běží nad neprázdným vstupním korpusem a kontroluje, že do výstupu
neuniká obsah buněk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.domain.synthetic_mentions import MentionVocabulary, mention_vocabulary
from mediparse.domain.synthetic_tables import (
    DEFAULT_ROWS,
    SyntheticRun,
    synthetic_rows,
)
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.diagnosis_candidates_file import (
    load_diagnosis_codes,
    load_diagnosis_keywords,
    load_text_normalization,
)
from mediparse.infrastructure.duckdb_cohort import MIN_CHARS
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker
from mediparse.infrastructure.text_mentions_file import load_text_mentions

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """Zapíše syntetický parquet pro každou tabulku manifestu a značku syntetiky.

    Returns:
        OK po zápisu; REFUSED pro chybějící nebo neplatný manifest či config, pro diagnózu
        bloku bez předpony a pro lexikon bez rozvržení kolem BHC.
    """
    args = _parser().parse_args(argv)
    shapes = read_manifest(args.manifest)
    synthetic = SyntheticRun(
        prefix=args.prefix,
        rows=args.rows,
        text_chars=MIN_CHARS,
        codes=load_diagnosis_codes(args.candidates),
        lexicon=load_section_lexicon(args.lexicon),
        mentions=_vocabulary(args.candidates, args.mentions),
    )
    # Řádky vzniknou všechny před prvním zápisem, aby odmítnutí nenechalo napůl zapsaný adresář.
    tables = {
        table: synthetic_rows(table, shape, synthetic)
        for table, shape in shapes.items()
    }
    for table, rows in tables.items():
        write_string_table(args.out / f"{table}.parquet", shapes[table].columns, rows)
    write_marker(args.out, SyntheticMarker(prefix=args.prefix, rows=args.rows))
    logger.info("%d syntetických tabulek v %s", len(shapes), args.out)
    return ExitCode.OK


def _vocabulary(candidates: Path, mentions: Path) -> MentionVocabulary:
    return mention_vocabulary(
        load_diagnosis_keywords(candidates),
        load_text_mentions(mentions),
        load_text_normalization(candidates),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-eda-synthetic-tables",
        description="Syntetické tabulky ve tvaru MIMIC pro render EDA.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    return parser
