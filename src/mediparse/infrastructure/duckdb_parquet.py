"""Společné části spojení DuckDB nad parquetem: pohled a umístění spillu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import duckdb

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from pathlib import Path

# Spill leží vedle dat pod resources/mimic, kam míří deny pravidla pro dočasná data.
SPILL_DIRECTORY: Final = "duckdb-tmp"


def create_parquet_view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    """Založí pohled ``name`` nad souborem parquet.

    Args:
        con: Spojení DuckDB.
        name: Jméno pohledu.
        path: Soubor parquet.

    Raises:
        InvalidInputError: Soubor nejde číst.
    """
    try:
        con.read_parquet(str(path)).create_view(name)
    except duckdb.Error as error:
        msg = f"Tabulka {path} nejde číst: {type(error).__name__}."
        # Text výjimky nese hodnoty buněk z MIMIC; nesmí do hlášky ani do logu, a proto
        # jen její typ a ``from None``.
        raise InvalidInputError(msg) from None
