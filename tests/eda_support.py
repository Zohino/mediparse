"""Společné fixtury testů databáze EDA: tabulky parquet a značka syntetiky."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_NAMES = ("discharge", "admissions")


def write_tables(directory: Path, names: tuple[str, ...] = DEFAULT_NAMES) -> Path:
    """Zapíše jednořádkové tabulky parquet do adresáře.

    Returns:
        Adresář s tabulkami.
    """
    for name in names:
        write_string_table(directory / f"{name}.parquet", ["a"], [{"a": "x"}])
    return directory


def mark_synthetic(directory: Path) -> Path:
    """Přidá do adresáře značku syntetiky.

    Returns:
        Týž adresář.
    """
    write_marker(directory, SyntheticMarker(prefix="t", rows=1))
    return directory
