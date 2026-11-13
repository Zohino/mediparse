"""Hranice souborů parquet: čtení se schématem, zápis s jednotnou kompresí a převod hodnot sloupců na doménové typy; chyby překládá na InvalidInputError."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.note import NoteIdentityError

COMPRESSION: Final = "zstd"


def read_table(path: Path, schema: pa.Schema) -> pa.Table:
    """Přečte celý soubor parquet a ověří, že má přesně dané schéma.

    Returns:
        Tabulka se schématem `schema`.

    Raises:
        InvalidInputError: Soubor neexistuje, není parquet nebo má jiné schéma.
    """
    with _parquet_errors(path):
        table = pq.read_table(path)
        _ensure_schema(path, table.schema, schema)
    return table


def iter_batches(
    path: Path, schema: pa.Schema, *, columns: list[str] | None = None, batch_size: int
) -> Iterator[pa.RecordBatch]:
    """Proudově přečte dávky a před projekcí sloupců ověří celé schéma souboru.

    Yields:
        Dávky v pořadí souboru, případně omezené na požadované sloupce.

    Raises:
        InvalidInputError: Soubor neexistuje, není parquet nebo má jiné schéma.
    """
    with _parquet_errors(path), pq.ParquetFile(path) as source:
        _ensure_schema(path, source.schema_arrow, schema)
        yield from source.iter_batches(batch_size=batch_size, columns=columns)


@contextmanager
def _parquet_errors(path: Path) -> Generator[None]:
    try:
        yield
    except FileNotFoundError as error:
        msg = f"Soubor {path} neexistuje."
        raise InvalidInputError(msg) from error
    except pa.ArrowInvalid as error:
        msg = f"Soubor {path} neodpovídá schématu: {error}"
        raise InvalidInputError(msg) from error


def _ensure_schema(path: Path, actual: pa.Schema, expected: pa.Schema) -> None:
    if not actual.equals(expected):
        msg = f"Soubor {path} neodpovídá schématu: čekané sloupce {', '.join(expected.names)}."
        raise InvalidInputError(msg)


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
    except ValueError:
        msg = f"Soubor {path} obsahuje neznámou diagnózu ve sloupci diagnosis."
        raise InvalidInputError(msg) from None


def note_identity_error(path: Path, error: NoteIdentityError) -> InvalidInputError:
    """Chyba identity zprávy řádku souboru jako chyba vstupu bez identifikátorů.

    Returns:
        Chyba s cestou a typem původní výjimky; volající ji vyhodí ``from None``.
    """
    # Text výjimky nese note_id, tedy subject_id pacienta MIMIC; nesmí do hlášky
    # ani do logu, a proto jen její typ.
    msg = (
        f"Soubor {path} obsahuje zprávu, jejíž note_id nepatří "
        f"jejímu subject_id: {type(error).__name__}."
    )
    return InvalidInputError(msg)


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
