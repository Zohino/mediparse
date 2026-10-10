"""Vstupní bod syntetických tabulek: jeden parquet na tabulku manifestu a značka syntetiky."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pyarrow.parquet as pq

from mediparse.entrypoints.eda_synthetic_tables import run
from mediparse.entrypoints.exit_code import ExitCode

if TYPE_CHECKING:
    from pathlib import Path

ROWS: Final = 12
MANIFEST = {
    "tables": [
        {"file": "discharge.csv.gz", "bytes": 1, "records": 3, "columns": ["a", "b"]},
        {"file": "admissions.csv.gz", "bytes": 1, "records": 3, "columns": ["c"]},
    ]
}


def _manifest(root: Path) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return path


def test_writes_one_all_string_parquet_per_manifest_table(tmp_path: Path) -> None:
    """Každá tabulka manifestu dostane parquet se samými řetězci a požadovaným počtem řádků."""
    out = tmp_path / "synthetic" / "tables"

    code = run(manifest=_manifest(tmp_path), out=out, prefix="canary", rows=ROWS)

    assert code is ExitCode.OK
    discharge = pq.read_table(out / "discharge.parquet")
    assert discharge.schema.names == ["a", "b"]
    assert {str(field.type) for field in discharge.schema} == {"string"}
    assert discharge.num_rows == ROWS
    assert pq.read_table(out / "admissions.parquet").num_rows == ROWS


def test_writes_marker_with_prefix_and_rows(tmp_path: Path) -> None:
    """Značka synthetic.json nese prefix a počet řádků."""
    out = tmp_path / "tables"

    run(manifest=_manifest(tmp_path), out=out, prefix="canary", rows=ROWS)

    marker = json.loads((out / "synthetic.json").read_text(encoding="utf-8"))
    assert marker == {"prefix": "canary", "rows": ROWS}


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    """Chybějící manifest ukončí krok kódem REFUSED a nic nezapíše."""
    out = tmp_path / "tables"

    code = run(manifest=tmp_path / "none.json", out=out, prefix="canary", rows=ROWS)

    assert code is ExitCode.REFUSED
    assert not out.exists()
