"""Čtení souboru parquet s pevným schématem: chyby souboru překládá na doménovou výjimku."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

COMPRESSION: Final = "zstd"


def read_table(path: Path, schema: pa.Schema) -> pa.Table:
    """Přečte celý soubor parquet a ověří, že má přesně dané schéma.

    Returns:
        Tabulka se schématem `schema`.

    Raises:
        InvalidInputError: Soubor neexistuje, není parquet nebo má jiné schéma.
    """
    try:
        table = pq.read_table(path)
    except FileNotFoundError as error:
        msg = f"Soubor {path} neexistuje."
        raise InvalidInputError(msg) from error
    except pa.ArrowInvalid as error:
        msg = f"Soubor {path} neodpovídá schématu: {error}"
        raise InvalidInputError(msg) from error
    if not table.schema.equals(schema):
        msg = f"Soubor {path} neodpovídá schématu: čekané sloupce {', '.join(schema.names)}."
        raise InvalidInputError(msg)
    return table


def write_table(
    path: Path, columns: Mapping[str, Sequence[object]], schema: pa.Schema
) -> None:
    """Zapíše sloupce se schématem do souboru parquet a vytvoří chybějící adresáře.

    Args:
        path: Cíl zápisu.
        columns: Hodnoty sloupců podle jména.
        schema: Schéma souboru, podle kterého se hodnoty převedou.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(dict(columns), schema=schema),
        path,
        compression=COMPRESSION,
    )


def parse_diagnosis(path: Path, value: str) -> Diagnosis:
    """Převede text ze sloupce souboru na diagnózu.

    Returns:
        Diagnóza téhož jména.

    Raises:
        InvalidInputError: Hodnota není diagnóza.
    """
    try:
        return Diagnosis(value)
    except ValueError as error:
        msg = f"Soubor {path} obsahuje neznámou diagnózu {value!r}."
        raise InvalidInputError(msg) from error


def write_string_table(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str | None]]
) -> None:
    """Zapíše soubor parquet se samými řetězcovými sloupci a vytvoří chybějící adresáře.

    Args:
        path: Cíl zápisu.
        columns: Sloupce v pořadí souboru.
        rows: Řádky jako slovníky sloupec na hodnotu.
    """
    schema = pa.schema([pa.field(column, pa.string()) for column in columns])
    write_table(
        path, {column: [row.get(column) for row in rows] for column in columns}, schema
    )
