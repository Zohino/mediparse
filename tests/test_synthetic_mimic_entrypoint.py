"""Kohorta nad syntetickými tabulkami má labely z nezávisle přečtených plánů."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.entrypoints import cohort
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, read_marker
from mediparse.infrastructure.table_provenance import provenance_path
from mediparse.synthetic.entrypoints import synthetic_mimic
from mediparse.synthetic.infrastructure.plans_file import PlansFile
from tests.support import (
    REPOSITORY_CORPUS,
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_INPUT_CORPUS,
    REPOSITORY_MIMIC_MANIFEST,
    REPOSITORY_PLANS,
    fixed_revision,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlannedNote

EXPECTED_NOTES: Final = 400


def test_tables_and_artifacts_have_synthetic_markers(
    tmp_path: Path, synthetic_tables: tuple[Path, Path]
) -> None:
    """Tabulky i odvozené artefakty nesou značku bez vymyšleného prefixu."""
    discharge, _ = synthetic_tables
    for directory in (discharge.parent, tmp_path):
        assert read_marker(directory) == SyntheticMarker(
            prefix=None, rows=EXPECTED_NOTES
        )


def test_cohort_matches_plans_and_source_hashes(
    tmp_path: Path, synthetic_tables: tuple[Path, Path]
) -> None:
    """Kohorta zachová 400 zpráv v pořadí note_id, celé řádky i pravdivou provenance."""
    discharge, diagnoses = synthetic_tables
    output = tmp_path / "cohort.parquet"
    manifest = tmp_path / "cohort" / "manifest.json"
    assert (
        cohort.run(
            environ={},
            discharge=discharge,
            diagnoses=diagnoses,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            corpus_rule=REPOSITORY_INPUT_CORPUS,
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
    expected = {plan["note_id"]: plan for plan in plans}
    positives = {
        "diabetes": 94,
        "atrial_fibrillation": 72,
        "ckd": 71,
        "copd": 31,
        "pneumonia": 28,
    }
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == EXPECTED_NOTES
    assert [row["note_id"] for row in rows] == sorted(plan["note_id"] for plan in plans)
    for row in rows:
        plan = expected[row["note_id"]]
        assert row["subject_id"] == plan["subject_id"]
        assert row["language"] == "en"
        assert row["text"] == (
            REPOSITORY_CORPUS / "en" / f"{row['note_id']}.txt"
        ).read_text(encoding="utf-8")
        assert {diagnosis for diagnosis in positives if row[diagnosis]} == set(
            plan["labels"]
        )
    assert {
        diagnosis: sum(row[diagnosis] for row in rows) for diagnosis in positives
    } == positives
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["invalid_codes"] == 0
    assert written["sources"] == {
        path.stem: {
            "source": f"https://mediparse.invalid/synthetic/{path.stem}",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (discharge, diagnoses)
    }


def test_tables_have_string_columns_and_synthetic_urls(
    synthetic_tables: tuple[Path, Path],
) -> None:
    """Vstup odpovídá typům MIMIC a URL nemohou vést na skutečný zdroj dat."""
    discharge, diagnoses = synthetic_tables
    for path in (discharge, diagnoses):
        assert all(field.type == pa.string() for field in pq.read_schema(path))
    for path in (discharge, diagnoses):
        written = json.loads(provenance_path(path).read_text(encoding="utf-8"))
        assert written == {
            "source": f"https://mediparse.invalid/synthetic/{path.stem}",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert not (discharge.parent.parent / "tables.json").exists()


def test_unpaired_corpus_is_refused_before_writing(
    tmp_path: Path, planned_note: PlannedNote, caplog: pytest.LogCaptureFixture
) -> None:
    """Plán bez zprávy odmítne generátor s důvodem a bez kteréhokoli výstupu."""
    plans = tmp_path / "plans.jsonl"
    PlansFile(plans).save([planned_note.plan])
    discharge = tmp_path / "discharge.parquet"
    diagnoses = tmp_path / "diagnoses_icd.parquet"

    assert (
        synthetic_mimic.run(
            plans=plans,
            corpus=tmp_path,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            manifest=REPOSITORY_MIMIC_MANIFEST,
            discharge=discharge,
            diagnoses=diagnoses,
            artifacts=tmp_path,
        )
        is ExitCode.REFUSED
    )
    assert not discharge.exists()
    assert not diagnoses.exists()
    assert not provenance_path(discharge).exists()
    assert not provenance_path(diagnoses).exists()
    assert f"Plány bez zprávy v jazyce en: {planned_note.plan.note_id}" in caplog.text


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
            artifacts=output,
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
            artifacts=tmp_path,
        )
        is ExitCode.OK
    )

    assert pq.read_schema(tmp_path / f"{table}.parquet").names == selected["columns"]
