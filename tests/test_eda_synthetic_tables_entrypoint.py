"""Vstupní bod syntetických tabulek: jeden parquet na tabulku manifestu a značka syntetiky."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pyarrow.parquet as pq

from mediparse.entrypoints.eda_synthetic_tables import DEFAULT_ROWS, run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.duckdb_cohort import MIN_CHARS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

ROWS: Final = 3
MANIFEST = {
    "tables": [
        {"file": "discharge.csv.gz", "bytes": 1, "records": 3, "columns": ["a", "b"]},
        {"file": "admissions.csv.gz", "bytes": 1, "records": 3, "columns": ["c"]},
    ]
}


def _manifest(root: Path, manifest: Mapping[str, object] = MANIFEST) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _arguments(manifest: Path, out: Path, *, rows: int | None) -> list[str]:
    arguments = [f"--manifest={manifest}", f"--out={out}", "--prefix=canary"]
    if rows is not None:
        arguments.append(f"--rows={rows}")
    return arguments


def test_writes_one_all_string_parquet_per_manifest_table(tmp_path: Path) -> None:
    """Každá tabulka manifestu dostane parquet se samými řetězci a požadovaným počtem řádků."""
    out = tmp_path / "synthetic" / "tables"

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS))

    assert code is ExitCode.OK
    discharge = pq.read_table(out / "discharge.parquet")
    assert discharge.schema.names == ["a", "b"]
    assert {str(field.type) for field in discharge.schema} == {"string"}
    assert discharge.num_rows == ROWS
    assert pq.read_table(out / "admissions.parquet").num_rows == ROWS


def test_writes_marker_with_prefix_and_rows(tmp_path: Path) -> None:
    """Značka synthetic.json nese prefix a počet řádků."""
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path), out, rows=ROWS))

    marker = json.loads((out / "synthetic.json").read_text(encoding="utf-8"))
    assert marker == {"prefix": "canary", "rows": ROWS}


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    """Chybějící manifest ukončí krok kódem REFUSED a nic nezapíše."""
    out = tmp_path / "tables"

    code = run(_arguments(tmp_path / "none.json", out, rows=ROWS))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_rows_default_when_not_given(tmp_path: Path) -> None:
    """Bez ``--rows`` se zapíše ``DEFAULT_ROWS`` řádků."""
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path), out, rows=None))

    assert pq.read_table(out / "admissions.parquet").num_rows == DEFAULT_ROWS


def test_notes_join_codes_and_pass_the_length_filter(tmp_path: Path) -> None:
    """Zprávy sdílí hadm_id s kódy ICD a mají aspoň MIN_CHARS znaků, takže vstupní korpus není prázdný."""
    manifest = {
        "tables": [
            {
                "file": "discharge.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["hadm_id", "text"],
            },
            {
                "file": "diagnoses_icd.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["hadm_id"],
            },
        ]
    }
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path, manifest), out, rows=ROWS))

    notes = pq.read_table(out / "discharge.parquet").to_pydict()
    codes = pq.read_table(out / "diagnoses_icd.parquet").to_pydict()
    assert notes["hadm_id"] == codes["hadm_id"]
    assert min(len(text) for text in notes["text"]) >= MIN_CHARS
