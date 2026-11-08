"""Vstupní bod převodu tabulky MIMIC: skutečné soubory v adresáři testu."""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import TYPE_CHECKING

from mediparse.domain.mimic_table import TableProvenance
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.mimic_parquet import run
from mediparse.infrastructure.table_provenance import provenance_path

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

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
            environ={},
            table=table,
            inventory=inventory,
            parquet=parquet,
        )
        == ExitCode.OK
    )
    assert parquet.is_file()
    written = TableProvenance.model_validate_json(
        parquet.with_suffix(".source.json").read_bytes()
    )
    assert written == TableProvenance(
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
            environ={},
            table=table,
            inventory=inventory,
            parquet=parquet,
        )
        == ExitCode.REFUSED
    )


def test_csv_changed_since_inventory_is_refused_without_provenance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Otisk jiného CSV v inventáři nesmí být původem nové vazby."""
    table, inventory = _files(tmp_path, RECORDS)
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    other_csv = gzip.compress(b"a,b\n3,y\n4,\n", mtime=0)
    payload["sha256"] = hashlib.sha256(other_csv).hexdigest()
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    parquet = tmp_path / "t.parquet"

    assert (
        run(environ={}, table=table, inventory=inventory, parquet=parquet)
        is ExitCode.REFUSED
    )
    assert not provenance_path(parquet).exists()
    assert caplog.messages == [
        f"Soubor {table.name} se od validace změnil; spusťte validaci znovu."
    ]
