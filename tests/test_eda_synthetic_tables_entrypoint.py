"""Vstupní bod syntetických tabulek: jeden parquet na tabulku manifestu a značka syntetiky."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import duckdb
import pyarrow.parquet as pq
import pytest

from mediparse.domain.cohort import MIN_CELL
from mediparse.domain.synthetic_tables import DEFAULT_ROWS
from mediparse.entrypoints.eda_synthetic_tables import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.duckdb_cohort import MIN_CHARS
from tests.cohort_support import candidate_prefixes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

ROWS: Final = 3
ARGPARSE_USAGE_ERROR: Final = 2
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


def _candidates(root: Path, prefixes: Mapping[str, object] | None = None) -> Path:
    path = root / "candidates.json"
    path.write_text(
        json.dumps({"candidates": prefixes or candidate_prefixes()}),
        encoding="utf-8",
    )
    return path


def _arguments(
    manifest: Path, out: Path, *, rows: int | None, candidates: Path | None = None
) -> list[str]:
    arguments = [
        f"--manifest={manifest}",
        f"--candidates={candidates or _candidates(manifest.parent)}",
        f"--out={out}",
        "--prefix=canary",
    ]
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


def test_marker_carries_default_rows_from_domain(tmp_path: Path) -> None:
    """Bez ``--rows`` značka nese ``DEFAULT_ROWS`` z domény."""
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path), out, rows=None))

    marker = json.loads((out / "synthetic.json").read_text(encoding="utf-8"))
    assert marker["rows"] == DEFAULT_ROWS


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


def test_missing_candidates_argument_exits_with_usage_error(tmp_path: Path) -> None:
    """Bez ``--candidates`` argparse skončí kódem 2."""
    with pytest.raises(SystemExit) as raised:
        run([f"--manifest={_manifest(tmp_path)}", f"--out={tmp_path}", "--prefix=p"])

    assert raised.value.code == ARGPARSE_USAGE_ERROR


def test_invalid_candidates_are_refused_without_output(tmp_path: Path) -> None:
    """Neplatný config skončí REFUSED a nevznikne žádný parquet."""
    out = tmp_path / "tables"
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS, candidates=broken))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_candidates_without_icd9_prefix_of_block_diagnosis_are_refused(
    tmp_path: Path,
) -> None:
    """Config platný pro loader, ale bez předpony ICD-9 u diabetu, skončí REFUSED bez parquetu."""
    prefixes = candidate_prefixes()
    prefixes["diabetes"]["icd9_prefixes"] = []
    prefixes["diabetes"]["icd10_prefixes"] = ["E11"]
    out = tmp_path / "tables"

    code = run(
        _arguments(
            _manifest(tmp_path),
            out,
            rows=ROWS,
            candidates=_candidates(tmp_path, prefixes),
        )
    )

    assert code is ExitCode.REFUSED
    assert not out.exists()


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
                "columns": ["hadm_id", "icd_version"],
            },
        ]
    }
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path, manifest), out, rows=None))

    notes = pq.read_table(out / "discharge.parquet").to_pydict()
    codes = pq.read_table(out / "diagnoses_icd.parquet").to_pydict()
    assert notes["hadm_id"] == codes["hadm_id"]
    assert min(len(text) for text in notes["text"]) >= MIN_CHARS
    assert len(notes["text"]) == DEFAULT_ROWS
    with duckdb.connect() as connection:
        joined = connection.execute(
            "SELECT TRY_CAST(c.icd_version AS INTEGER), count(*) "
            "FROM read_parquet(?) n JOIN read_parquet(?) c USING (hadm_id) "
            "GROUP BY 1",
            [str(out / "discharge.parquet"), str(out / "diagnoses_icd.parquet")],
        ).fetchall()
    assert dict(joined) == {
        9: 2 * MIN_CELL,
        10: 2 * MIN_CELL,
        None: DEFAULT_ROWS - 4 * MIN_CELL,
    }
