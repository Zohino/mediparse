"""Vstupní bod syntetických tabulek pro render EDA: tabulky ve tvaru MIMIC bez dat MIMIC.

Konzolový skript ``mediparse-eda-synthetic-tables`` čte jen veřejný manifest
validace a zapíše pro každou jeho tabulku parquet se samými řetězci, kde je každá
buňka jedinečná. Render dokumentů nad nimi slouží ke kontrole, že do výstupu
neuniká obsah buněk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.synthetic_tables import synthetic_rows
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_ROWS: Final = 12


def main() -> ExitCode:
    """Konzolový skript ``mediparse-eda-synthetic-tables``.

    Returns:
        Návratový kód.
    """
    configure_logging()
    return _command(sys.argv[1:])


def _command(argv: Sequence[str]) -> ExitCode:
    args = _parser().parse_args(argv)
    return run(manifest=args.manifest, out=args.out, prefix=args.prefix, rows=args.rows)


@refusing_invalid_input
def run(
    *, manifest: Path, out: Path, prefix: str, rows: int = DEFAULT_ROWS
) -> ExitCode:
    """Zapíše syntetický parquet pro každou tabulku manifestu a značku syntetiky.

    Returns:
        OK po zápisu; REFUSED pro chybějící nebo neplatný manifest.
    """
    shapes = read_manifest(manifest)
    for table, shape in shapes.items():
        write_string_table(
            out / f"{table}.parquet",
            shape.columns,
            synthetic_rows(table, shape, prefix, rows),
        )
    write_marker(out, SyntheticMarker(prefix=prefix, rows=rows))
    logger.info("%d syntetických tabulek v %s", len(shapes), out)
    return ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-eda-synthetic-tables",
        description="Syntetické tabulky ve tvaru MIMIC pro render EDA.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    return parser
