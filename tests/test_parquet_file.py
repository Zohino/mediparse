"""Čtení parquet s pevným schématem: každá chyba souboru je doménová chyba vstupu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.parquet_file import read_table, write_string_table

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = pa.schema([pa.field("note_id", pa.string(), nullable=False)])


def test_reads_table_with_schema(tmp_path: Path) -> None:
    """Soubor s pevným schématem se přečte celý."""
    path = tmp_path / "table.parquet"
    expected = pa.Table.from_pydict({"note_id": ["a", "b"]}, schema=SCHEMA)
    pq.write_table(expected, path)

    assert read_table(path, SCHEMA).equals(expected)


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor se ohlásí jménem."""
    with pytest.raises(InvalidInputError, match=r"missing\.parquet neexistuje"):
        read_table(tmp_path / "missing.parquet", SCHEMA)


def test_non_parquet_file_is_invalid_input(tmp_path: Path) -> None:
    """Soubor, který není parquet, se ohlásí jménem."""
    path = tmp_path / "text.parquet"
    path.write_text("ne parquet", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"text\.parquet neodpovídá schématu"):
        read_table(path, SCHEMA)


def test_foreign_schema_names_expected_columns(tmp_path: Path) -> None:
    """Parquet s cizím schématem se ohlásí jménem a čekanými sloupci."""
    path = tmp_path / "foreign.parquet"
    pq.write_table(pa.table({"x": [1]}), path)

    with pytest.raises(
        InvalidInputError, match=r"foreign\.parquet neodpovídá schématu.*note_id"
    ):
        read_table(path, SCHEMA)


def test_written_string_table_reads_back_with_all_string_schema(tmp_path: Path) -> None:
    """Zapsaná tabulka má samé řetězcové sloupce v pořadí a čte se zpět."""
    path = tmp_path / "out" / "table.parquet"

    write_string_table(path, ("b", "a"), [{"b": "1", "a": "2"}, {"b": "3", "a": "4"}])

    table = pq.read_table(path)
    assert table.schema.names == ["b", "a"]
    assert {str(field.type) for field in table.schema} == {"string"}
    assert table.to_pydict() == {"b": ["1", "3"], "a": ["2", "4"]}
