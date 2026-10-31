"""Kohorta nad syntetickými tabulkami má labely z nezávisle přečtených plánů."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.entrypoints import cohort, synthetic_mimic
from mediparse.entrypoints.exit_code import ExitCode
from tests.support import (
    REPOSITORY_CORPUS,
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_MIMIC_MANIFEST,
    REPOSITORY_PLANS,
    fixed_revision,
)

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_NOTES: Final = 400


@pytest.fixture
def synthetic_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Zapíše obě vstupní tabulky a otisky skutečným entrypointem.

    Returns:
        Cesty k discharge, diagnoses_icd a configu otisků.
    """
    discharge = tmp_path / "mimic" / "parquet" / "discharge.parquet"
    diagnoses = discharge.with_name("diagnoses_icd.parquet")
    tables = tmp_path / "mimic" / "tables.json"
    assert (
        synthetic_mimic.run(
            plans=REPOSITORY_PLANS,
            corpus=REPOSITORY_CORPUS,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            manifest=REPOSITORY_MIMIC_MANIFEST,
            discharge=discharge,
            diagnoses=diagnoses,
            tables=tables,
        )
        is ExitCode.OK
    )
    return discharge, diagnoses, tables


def test_cohort_matches_plans_and_source_hashes(
    tmp_path: Path, synthetic_tables: tuple[Path, Path, Path]
) -> None:
    """Skutečná kohorta zachová všech 400 plánů i jejich labely a pravdivou provenance."""
    discharge, diagnoses, tables = synthetic_tables
    output = tmp_path / "cohort.parquet"
    manifest = tmp_path / "cohort" / "manifest.json"
    assert (
        cohort.run(
            discharge=discharge,
            diagnoses=diagnoses,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            tables=tables,
            cohort=output,
            manifest=manifest,
            revision=fixed_revision,
        )
        is ExitCode.OK
    )

    plans = [
        json.loads(line)
        for line in REPOSITORY_PLANS.read_text(encoding="utf-8").splitlines()
    ]
    expected = {plan["note_id"]: set(plan["labels"]) for plan in plans}
    positives = {
        "diabetes": 94,
        "atrial_fibrillation": 72,
        "ckd": 71,
        "copd": 31,
        "pneumonia": 28,
    }
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == EXPECTED_NOTES
    assert {row["note_id"] for row in rows} == set(expected)
    for row in rows:
        assert {diagnosis for diagnosis in positives if row[diagnosis]} == expected[
            row["note_id"]
        ]
    assert {
        diagnosis: sum(row[diagnosis] for row in rows) for diagnosis in positives
    } == positives
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["invalid_codes"] == 0
    assert written["sources"] == {
        "discharge.csv.gz": hashlib.sha256(discharge.read_bytes()).hexdigest(),
        "diagnoses_icd.csv.gz": hashlib.sha256(diagnoses.read_bytes()).hexdigest(),
    }


def test_tables_have_string_columns_and_synthetic_urls(
    synthetic_tables: tuple[Path, Path, Path],
) -> None:
    """Vstup odpovídá typům MIMIC a URL nemohou vést na skutečný zdroj dat."""
    discharge, diagnoses, tables = synthetic_tables
    for path in (discharge, diagnoses):
        assert all(field.type == pa.string() for field in pq.read_schema(path))
    written = json.loads(tables.read_text(encoding="utf-8"))
    assert [table["url"] for table in written["mimic_tables"]] == [
        "https://mediparse.invalid/synthetic/discharge.csv.gz",
        "https://mediparse.invalid/synthetic/diagnoses_icd.csv.gz",
    ]


def test_missing_manifest_is_refused_before_writing(tmp_path: Path) -> None:
    """Chybějící tvar vstupu se odmítne bez částečného výstupu."""
    output = tmp_path / "output"
    assert (
        synthetic_mimic.run(
            plans=REPOSITORY_PLANS,
            corpus=REPOSITORY_CORPUS,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            manifest=tmp_path / "missing.json",
            discharge=output / "discharge.parquet",
            diagnoses=output / "diagnoses_icd.parquet",
            tables=output / "tables.json",
        )
        is ExitCode.REFUSED
    )
    assert not output.exists()


@pytest.mark.parametrize("table", ["discharge", "diagnoses_icd"])
def test_written_columns_follow_manifest(tmp_path: Path, table: str) -> None:
    """Zapisovač dodrží pořadí manifestu i tehdy, když se liší od pořadí klíčů řádku."""
    inventory = json.loads(REPOSITORY_MIMIC_MANIFEST.read_text(encoding="utf-8"))
    selected = next(
        item for item in inventory["tables"] if item["file"] == f"{table}.csv.gz"
    )
    selected["columns"].reverse()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(inventory), encoding="utf-8")
    discharge = tmp_path / "discharge.parquet"
    diagnoses = tmp_path / "diagnoses_icd.parquet"

    assert (
        synthetic_mimic.run(
            plans=REPOSITORY_PLANS,
            corpus=REPOSITORY_CORPUS,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            manifest=manifest,
            discharge=discharge,
            diagnoses=diagnoses,
            tables=tmp_path / "tables.json",
        )
        is ExitCode.OK
    )

    assert pq.read_schema(tmp_path / f"{table}.parquet").names == selected["columns"]
