"""Čtení parquet s pevným schématem: každá chyba souboru je doménová chyba vstupu."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.parquet_file import (
    iter_batches,
    parse_diagnosis,
    read_table,
    write_string_table,
    write_table,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
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


def test_parse_diagnosis_names_path_and_column_without_value(tmp_path: Path) -> None:
    """Neznámá diagnóza ohlásí cestu a sloupec bez hodnoty buňky a kontextu."""
    path = tmp_path / "t.parquet"
    value = "tajny-text-123"

    with pytest.raises(InvalidInputError) as raised:
        parse_diagnosis(path, value)

    assert value not in str(raised.value)
    assert str(raised.value) == (
        f"Soubor {path} obsahuje neznámou diagnózu ve sloupci diagnosis."
    )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_batches_preserve_rows_and_project_columns(tmp_path: Path) -> None:
    """Pět řádků se čte v dávkách 2, 2, 1 a promítne se jen požadovaný sloupec."""
    path = tmp_path / "batches.parquet"
    schema = pa.schema([("note_id", pa.string()), ("count", pa.int64())])
    pq.write_table(
        pa.table(
            {"note_id": ["a", "b", "c", "d", "e"], "count": [1, 2, 3, 4, 5]},
            schema=schema,
        ),
        path,
    )

    batches = list(iter_batches(path, schema, columns=["note_id"], batch_size=2))

    assert [batch.num_rows for batch in batches] == [2, 2, 1]
    assert pa.Table.from_batches(batches).to_pydict() == {
        "note_id": ["a", "b", "c", "d", "e"]
    }


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "schema"])
def test_batches_translate_file_errors_with_complete_message(
    tmp_path: Path, invalid: str
) -> None:
    """Proudové čtení hlásí celé doménové chyby stejně jako společná hranice souboru."""
    path = tmp_path / "table.parquet"
    messages = {
        "missing": f"Soubor {path} neexistuje.",
        "schema": f"Soubor {path} neodpovídá schématu: čekané sloupce note_id.",
        "corrupt": (
            f"Soubor {path} neodpovídá schématu: Parquet magic bytes not found in footer. "
            "Either the file is corrupted or this is not a parquet file."
        ),
    }
    if invalid == "schema":
        pq.write_table(pa.table({"note_id": ["a"], "other": [1]}), path)
    elif invalid == "corrupt":
        path.write_text("není parquet", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=f"^{re.escape(messages[invalid])}$"):
        list(iter_batches(path, SCHEMA, columns=["note_id"], batch_size=2))


def test_batches_translate_error_while_reading_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Překlad chyb pokryje i selhání po vydání první platné dávky."""
    path = tmp_path / "table.parquet"
    pq.write_table(pa.Table.from_pydict({"note_id": ["a"]}, schema=SCHEMA), path)

    def broken_batches(*args: object, **kwargs: object) -> Iterator[pa.RecordBatch]:
        del args, kwargs
        yield pa.record_batch([["a"]], schema=SCHEMA)
        msg = "Syntetická chyba další dávky."
        raise pa.ArrowInvalid(msg)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", broken_batches)
    batches = iter_batches(path, SCHEMA, batch_size=1)
    assert next(batches).column("note_id").to_pylist() == ["a"]
    message = f"Soubor {path} neodpovídá schématu: Syntetická chyba další dávky."
    with pytest.raises(InvalidInputError, match=f"^{re.escape(message)}$"):
        next(batches)
