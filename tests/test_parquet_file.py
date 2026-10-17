"""Čtení parquet s pevným schématem: každá chyba souboru je doménová chyba vstupu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.parquet_file import (
    parse_diagnosis,
    read_table,
    write_string_table,
    write_table,
)

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


def test_write_table_writes_columns_with_schema_and_zstd(tmp_path: Path) -> None:
    """Sloupce se zapíšou se schématem, chybějící adresář se založí a komprese je zstd."""
    path = tmp_path / "out" / "table.parquet"

    write_table(path, {"note_id": ["a", "b"]}, SCHEMA)

    assert read_table(path, SCHEMA).column("note_id").to_pylist() == ["a", "b"]
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


def test_parse_diagnosis_converts_known_value(tmp_path: Path) -> None:
    """Známá hodnota se převede na výčet."""
    assert parse_diagnosis(tmp_path / "t.parquet", "ckd") is Diagnosis.CKD


def test_parse_diagnosis_names_path_and_value_of_unknown(tmp_path: Path) -> None:
    """Neznámá diagnóza je chybný vstup, jehož hláška nese cestu i hodnotu."""
    with pytest.raises(
        InvalidInputError, match=r"t\.parquet obsahuje neznámou diagnózu 'flu'"
    ):
        parse_diagnosis(tmp_path / "t.parquet", "flu")
