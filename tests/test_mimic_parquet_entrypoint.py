"""Vstupní bod převodu tabulky MIMIC: skutečné soubory v adresáři testu."""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

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
            "bytes": table.stat().st_size,
            "records": records,
            "columns": ["a", "b"],
        })
    )
    return table, inventory


def test_matching_inventory_writes_parquet(tmp_path: Path) -> None:
    """Shoda s inventářem: OK a parquet existuje."""
    table, inventory = _files(tmp_path, RECORDS)
    parquet = tmp_path / "parquet" / "t.parquet"
    assert run(table=table, inventory=inventory, parquet=parquet) == ExitCode.OK
    assert parquet.is_file()


def test_count_mismatch_is_refused(tmp_path: Path) -> None:
    """Jiný počet než v inventáři: REFUSED."""
    table, inventory = _files(tmp_path, RECORDS + 1)
    parquet = tmp_path / "t.parquet"
    assert run(table=table, inventory=inventory, parquet=parquet) == ExitCode.REFUSED
