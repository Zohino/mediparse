"""Vstupní bod převodu tabulky MIMIC: skutečné soubory v adresáři testu."""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import TYPE_CHECKING

from mediparse.domain.mimic_table import ParquetSource
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.mimic_parquet import run

if TYPE_CHECKING:
    from pathlib import Path

RECORDS = 2


def _files(root: Path, records: int) -> tuple[Path, Path]:
    table = root / "t.csv.gz"
    table.write_bytes(gzip.compress(b"a,b\n1,x\n2,\n"))
    inventory = root / "t.csv.gz.json"
    inventory.write_text(
        json.dumps({
            "file": table.name,
            "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            "bytes": table.stat().st_size,
            "records": records,
            "columns": ["a", "b"],
        }),
        encoding="utf-8",
    )
    return table, inventory


def test_matching_inventory_writes_parquet(tmp_path: Path) -> None:
    """Soulad s inventářem: OK a parquet existuje."""
    table, inventory = _files(tmp_path, RECORDS)
    parquet = tmp_path / "parquet" / "t.parquet"
    assert (
        run(
            table=table,
            inventory=inventory,
            parquet=parquet,
            source=parquet.with_suffix(".source.json"),
        )
        == ExitCode.OK
    )
    assert parquet.is_file()
    written = ParquetSource.model_validate_json(
        parquet.with_suffix(".source.json").read_bytes()
    )
    assert written == ParquetSource(
        source=table.name,
        source_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
    )


def test_count_mismatch_is_refused(tmp_path: Path) -> None:
    """Jiný počet než v inventáři: REFUSED."""
    table, inventory = _files(tmp_path, RECORDS + 1)
    parquet = tmp_path / "t.parquet"
    assert (
        run(
            table=table,
            inventory=inventory,
            parquet=parquet,
            source=parquet.with_suffix(".source.json"),
        )
        == ExitCode.REFUSED
    )


def test_noncanonical_source_path_is_refused_before_conversion(tmp_path: Path) -> None:
    """Výstup pravidla musí souhlasit s cestou vazby, kterou pak čtou konzumenti."""
    table, inventory = _files(tmp_path, RECORDS)
    parquet = tmp_path / "t.parquet"
    source = tmp_path / "different.json"

    assert (
        run(table=table, inventory=inventory, parquet=parquet, source=source)
        == ExitCode.REFUSED
    )
    assert not parquet.exists()
    assert not source.exists()
