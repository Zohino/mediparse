"""Provenance renderu dokumentu EDA pro přílohy na knitru."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_mapping
from mediparse.infrastructure.mimic_tables import TABLES_PATH, load_mimic_tables
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.synthetic_marker import read_marker

if TYPE_CHECKING:
    from collections.abc import Mapping

LOCK_NAME: Final = "pixi.lock"
PACKAGES: Final = ("python", "duckdb")


def provenance(environ: Mapping[str, str], root: Path, parquet: Path) -> dict[str, str]:
    """Složí provenance renderu: revizi zdroje, otisk zámku, zdroj dat a verze.

    Args:
        environ: Proměnné prostředí; MEDIPARSE_COMMIT a MEDIPARSE_DIRTY mají přednost před gitem.
        root: Kořen repozitáře s ``pixi.lock`` a ``config/``.
        parquet: Adresář tabulek parquet, ze kterých dokument čte.

    Returns:
        Popisek a hodnota ve stabilním pořadí pro tabulku přílohy.
    """
    revision = revision_source(environ, root)()
    return {
        "commit": revision.commit,
        "dirty": str(revision.dirty).lower(),
        "pixi_lock_sha256": file_sha256(root / LOCK_NAME),
        **_data_source(root, parquet),
        **installed_versions(PACKAGES),
    }


def _data_source(root: Path, parquet: Path) -> dict[str, str]:
    marker = read_marker(parquet)
    if marker is not None:
        return {"data_source": f"syntetické tabulky (prefix {marker.prefix})"}
    tables = load_mimic_tables(root / TABLES_PATH)
    return {
        "data_source": "otisky CSV z config/mimic_tables.json",
        **{f"csv_sha256 {name}": table.sha256 for name, table in tables.items()},
    }


def run(root: Path, parquet: Path, out: Path, environ: Mapping[str, str]) -> None:
    """Zapíše provenance renderu do souboru pro nezávislé čtení v R."""
    write_mapping(out, provenance(environ, root, parquet))


def main() -> ExitCode:
    """Nastaví logování a načte cesty zdrojů a výstupu.

    Returns:
        OK po zápisu, REFUSED při odmítnutém vstupu nebo výstupu.
    """
    configure_logging()
    return _run_cli()


@refusing_invalid_input
def _run_cli() -> ExitCode:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "parquet", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.parquet, args.out, os.environ)
    return ExitCode.OK
