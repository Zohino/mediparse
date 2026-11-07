"""Provenance renderu dokumentu EDA pro přílohy na knitru."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import StringMapping, write_model
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.synthetic_marker import read_marker
from mediparse.infrastructure.table_provenance import verified_table_provenance

if TYPE_CHECKING:
    from collections.abc import Mapping

LOCK_NAME: Final = "pixi.lock"
PACKAGES: Final = ("python", "duckdb")


def provenance(environ: Mapping[str, str], root: Path, parquet: Path) -> dict[str, str]:
    """Složí provenance renderu: revizi zdroje, otisk zámku, zdroj dat a verze.

    Args:
        environ: Proměnné prostředí; MEDIPARSE_COMMIT a MEDIPARSE_DIRTY mají přednost před gitem.
        root: Kořen repozitáře s ``pixi.lock``.
        parquet: Adresář tabulek parquet, ze kterých dokument čte.

    Returns:
        Popisek a hodnota ve stabilním pořadí pro tabulku přílohy.
    """
    revision = revision_source(environ, root)()
    return {
        "commit": revision.commit,
        "dirty": str(revision.dirty).lower(),
        "pixi_lock_sha256": file_sha256(root / LOCK_NAME),
        **_data_source(parquet),
        **installed_versions(PACKAGES),
    }


def _data_source(parquet: Path) -> dict[str, str]:
    marker = read_marker(parquet)
    if marker is not None:
        return {"data_source": f"syntetické tabulky (prefix {marker.prefix})"}
    details = {"data_source": "tabulky parquet ověřené proti vazbě na otisky CSV"}
    for path in sorted(parquet.glob("*.parquet")):
        source = verified_table_provenance(path)
        details[f"csv_sha256 {path.stem}"] = source.source_sha256
        details[f"parquet_sha256 {path.stem}"] = source.sha256
    return details


def run(root: Path, parquet: Path, out: Path, environ: Mapping[str, str]) -> None:
    """Zapíše provenance renderu do souboru pro nezávislé čtení v R."""
    write_model(out, StringMapping(provenance(environ, root, parquet)))


def main() -> ExitCode:
    """Nastaví logování a načte cesty zdrojů a výstupu.

    Returns:
        OK po zápisu, REFUSED při odmítnutém vstupu.
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
