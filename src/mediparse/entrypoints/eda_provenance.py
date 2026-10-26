"""Provenance renderu dokumentu EDA pro přílohy na knitru i jupyteru."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.eda_provenance import provenance_rows
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.eda_database import ROOT_VARIABLE, parquet_dir
from mediparse.entrypoints.eda_export import export
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.mimic_tables import TABLES_PATH, load_mimic_tables
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.source_revision import revision_source
from mediparse.infrastructure.synthetic_marker import read_marker

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.infrastructure.thesis_export import MarkdownOutput

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


def appendix(environ: Mapping[str, str]) -> MarkdownOutput:
    """Zapíše a zobrazí tabulku provenance pro sešit na jupyteru.

    Args:
        environ: Prostředí renderu s kořenem projektu a proměnnými Quarta.

    Returns:
        Markdown přílohy s popiskem, id a šířkami sloupců.

    Raises:
        InvalidInputError: Chybí kořen projektu nebo proměnné Quarta.
    """
    root = environ.get(ROOT_VARIABLE)
    if not root:
        msg = f"Pro přílohu provenance nastavte {ROOT_VARIABLE}."
        raise InvalidInputError(msg)
    details = provenance(environ, Path(root), Path(parquet_dir(environ)))
    return export(environ).table(
        provenance_rows(details),
        header=("Položka", "Hodnota"),
        align="ll",
        label="tbl-provenance",
        caption="Provenance renderu.",
        widths=(28, 72),
    )
