"""Převod CSV na parquet: bezeztrátová kopie po blocích."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.infrastructure.csv_parquet import CsvParquetConversion

if TYPE_CHECKING:
    from pathlib import Path

MANY = 2000
TWO = 2
HEADER = "note_id,hadm_id,icd_code,text\n"
ROWS = '1,10,0389,"řádek\nzalomený ""v uvozovkách"""\n2,,V0251,krátký\n'


def _csv(root: Path, text: str) -> Path:
    path = root / "table.csv.gz"
    path.write_bytes(gzip.compress(text.encode()))
    return path


def _convert(root: Path, text: str, block_size: int = 1 << 20) -> TableShape:
    return CsvParquetConversion(
        _csv(root, text), root / "out" / "table.parquet", block_size
    ).convert()


def test_field_with_newline_and_quotes_survives(tmp_path: Path) -> None:
    """Zalomení a zdvojené uvozovky uvnitř pole zůstanou beze změny."""
    _convert(tmp_path, HEADER + ROWS)
    table = pq.read_table(tmp_path / "out" / "table.parquet")
    assert table.column("text")[0].as_py() == 'řádek\nzalomený "v uvozovkách"'
    assert table.num_rows == TWO


def test_empty_value_is_null(tmp_path: Path) -> None:
    """Prázdná hodnota je NULL."""
    _convert(tmp_path, HEADER + ROWS)
    table = pq.read_table(tmp_path / "out" / "table.parquet")
    assert table.column("hadm_id").to_pylist() == ["10", None]


def test_codes_stay_strings_with_leading_zero(tmp_path: Path) -> None:
    """Kód ICD zůstane řetězcem, včetně úvodní nuly."""
    _convert(tmp_path, HEADER + ROWS)
    table = pq.read_table(tmp_path / "out" / "table.parquet")
    assert table.column("icd_code").to_pylist() == ["0389", "V0251"]


def test_many_blocks_keep_count_and_string_schema(tmp_path: Path) -> None:
    """Víc bloků než jeden: počet řádků sedí a všechny sloupce jsou string."""
    body = "".join(f"{i},,{i:04d},text {i}\n" for i in range(MANY))
    shape = _convert(tmp_path, HEADER + body, block_size=1 << 12)
    parquet = pq.ParquetFile(tmp_path / "out" / "table.parquet")
    assert shape == TableShape(
        records=MANY, columns=("note_id", "hadm_id", "icd_code", "text")
    )
    assert parquet.metadata.num_rows == MANY
    assert parquet.metadata.num_row_groups > 1
    assert set(parquet.schema_arrow.types) == {pa.string()}
    assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"


def test_missing_file_is_refused(tmp_path: Path) -> None:
    """Chybějící soubor je chyba vstupu."""
    conversion = CsvParquetConversion(tmp_path / "none.csv.gz", tmp_path / "o.parquet")
    with pytest.raises(InvalidInputError, match="neexistuje"):
        conversion.convert()


def test_broken_gzip_is_refused(tmp_path: Path) -> None:
    """Soubor, který není gzip, je chyba vstupu."""
    source = tmp_path / "bad.csv.gz"
    source.write_bytes(b"not gzip at all")
    with pytest.raises(InvalidInputError, match="nejde převést"):
        CsvParquetConversion(source, tmp_path / "o.parquet").convert()


def test_truncated_gzip_is_refused(tmp_path: Path) -> None:
    """Useknutý gzip je chyba vstupu."""
    source = _csv(tmp_path, HEADER + ROWS * 500)
    source.write_bytes(source.read_bytes()[:-20])
    with pytest.raises(InvalidInputError, match="nejde převést"):
        CsvParquetConversion(source, tmp_path / "o.parquet").convert()


def test_unclosed_quotes_are_refused(tmp_path: Path) -> None:
    """Neuzavřené uvozovky jsou chyba vstupu."""
    with pytest.raises(InvalidInputError, match="uvozovkách"):
        _convert(tmp_path, HEADER + '1,10,0389,"nikdy neuzavřeno\n')


def test_multiline_fields_survive_across_blocks(tmp_path: Path) -> None:
    """Zalomení a zdvojené uvozovky v polích přežijí i hranice bloků."""
    body = "".join(f'{i},1,0389,"řádek {i}\nzalomený ""x"""\n' for i in range(MANY))
    shape = _convert(tmp_path, HEADER + body, block_size=1 << 12)
    parquet = pq.ParquetFile(tmp_path / "out" / "table.parquet")
    assert parquet.metadata.num_row_groups > 1
    assert shape.records == MANY
    table = parquet.read()
    assert table.num_rows == MANY
    assert table.column("text")[1234].as_py() == 'řádek 1234\nzalomený "x"'


def test_row_with_wrong_field_count_is_refused(tmp_path: Path) -> None:
    """Řádek s jiným počtem polí než hlavička je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="nejde převést"):
        _convert(tmp_path, "a,b\n1,2,3\n")


def test_header_not_utf8_is_refused(tmp_path: Path) -> None:
    """Hlavička, která není v UTF-8, je chyba vstupu."""
    source = tmp_path / "latin.csv.gz"
    source.write_bytes(gzip.compress(b"a\xff,b\n1,2\n"))
    with pytest.raises(InvalidInputError, match="nejde převést"):
        CsvParquetConversion(source, tmp_path / "o.parquet").convert()


def test_columns_come_from_written_schema(tmp_path: Path) -> None:
    """Sloupce jsou ty, které se zapsaly: BOM v hlavičce se do jména nedostane."""
    shape = _convert(tmp_path, "\ufeffa,b\n1,2\n")
    parquet = pq.ParquetFile(tmp_path / "out" / "table.parquet")
    assert shape.columns == tuple(parquet.schema_arrow.names) == ("a", "b")
    assert set(parquet.schema_arrow.types) == {pa.string()}


def test_duckdb_reads_converted_parquet(tmp_path: Path) -> None:
    """DuckDB přečte převod přes víc skupin řádků se stejným počtem, NULL i textem."""
    body = "".join(f'{i},,{i:04d},"řádek {i}\nzalomený ""x"""\n' for i in range(MANY))
    _convert(tmp_path, HEADER + body, block_size=1 << 12)
    path = tmp_path / "out" / "table.parquet"
    query = (
        "SELECT count(*), count(hadm_id), max(text) FILTER (icd_code = '1234') "
        "FROM read_parquet(?)"
    )
    row = duckdb.connect().execute(query, [str(path)]).fetchone()
    expected = (pq.ParquetFile(path).metadata.num_rows, 0, 'řádek 1234\nzalomený "x"')
    assert row == expected
