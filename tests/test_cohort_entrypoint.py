"""Vstupní bod kroku build_cohort: kohorta a manifest, odmítnutí malé buňky."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Final

from mediparse.domain.cohort import MIN_CELL, CohortManifest
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import SourceRevision
from mediparse.entrypoints import cohort
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from tests.cohort_support import icd, note, text, write_tables

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

REVISION: Final = SourceRevision(commit="a" * 40, dirty=False)
KEPT_NOTES: Final = 56
COPD_NOTES: Final = 12
DISCHARGE_SHA: Final = "1" * 64
DIAGNOSES_SHA: Final = "2" * 64
PREFIXES: Final = {
    "diabetes": (["250"], ["E11"]),
    "copd": (["496"], ["J44"]),
    "ckd": (["585"], ["N18"]),
    "pneumonia": (["486"], ["J18"]),
    "atrial_fibrillation": (["4273"], ["I48"]),
}


def _revision() -> SourceRevision:
    return REVISION


def _no_git() -> SourceRevision:
    msg = "git chybí"
    raise InvalidInputError(msg)


def _configs(root: Path) -> tuple[Path, Path]:
    candidates = root / "diagnosis_candidates.json"
    candidates.write_text(
        json.dumps({
            "candidates": {
                key: {"icd9_prefixes": nine, "icd10_prefixes": ten}
                for key, (nine, ten) in PREFIXES.items()
            }
        }),
        encoding="utf-8",
    )
    tables = root / "mimic_tables.json"
    base = "https://physionet.org/files/mimiciv/3.1/hosp"
    tables.write_text(
        json.dumps({
            "mimic_tables": [
                {"url": f"{base}/discharge.csv.gz", "sha256": DISCHARGE_SHA},
                {"url": f"{base}/diagnoses_icd.csv.gz", "sha256": DIAGNOSES_SHA},
            ]
        }),
        encoding="utf-8",
    )
    return candidates, tables


def test_writes_cohort_and_manifest(tmp_path: Path) -> None:
    """Kohorta a manifest vzniknou; otisky odpovídají souborům a configu tabulek."""
    discharge, diagnoses = write_tables(tmp_path)
    candidates, tables = _configs(tmp_path)
    out = tmp_path / "resources" / "cohort.parquet"
    manifest = tmp_path / "results" / "cohort" / "manifest.json"

    code = cohort.run(
        discharge=discharge,
        diagnoses=diagnoses,
        candidates=candidates,
        tables=tables,
        cohort=out,
        manifest=manifest,
        revision=_revision,
    )

    written = CohortManifest.model_validate_json(
        manifest.read_text(encoding="utf-8"), strict=False
    )
    assert code is ExitCode.OK
    assert written.cohort_sha256 == file_sha256(out)
    assert written.config_sha256 == file_sha256(candidates)
    assert written.sources == {
        "discharge.csv.gz": DISCHARGE_SHA,
        "diagnoses_icd.csv.gz": DIAGNOSES_SHA,
    }
    assert written.environment.source == REVISION
    assert set(written.environment.packages) == set(cohort.PACKAGES)
    assert "pydantic" in written.environment.packages
    assert written.invalid_codes == 1
    assert written.counts.funnel[-1].notes == KEPT_NOTES
    assert written.counts.positives[Diagnosis.COPD] == COPD_NOTES
    assert (out.parent / "duckdb-tmp").is_dir()


def test_cell_below_threshold_is_refused_and_nothing_is_written(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza pod deset zpráv krok odmítne; log ji jmenuje a nenese počty."""
    caplog.set_level(logging.DEBUG)
    discharge, diagnoses = write_tables(
        tmp_path,
        [note("n01", 1, 10, text("a"))],
        [icd(10, 1, "E119", "10")],
        padded=False,
    )
    candidates, tables = _configs(tmp_path)
    out = tmp_path / "cohort.parquet"
    manifest = tmp_path / "manifest.json"

    code = cohort.run(
        discharge=discharge,
        diagnoses=diagnoses,
        candidates=candidates,
        tables=tables,
        cohort=out,
        manifest=manifest,
        revision=_revision,
    )

    assert code is ExitCode.REFUSED
    assert any("diabetes" in message for message in caplog.messages)
    numbers = {n for message in caplog.messages for n in re.findall(r"\d+", message)}
    assert numbers <= {str(MIN_CELL)}
    assert not out.exists()
    assert not manifest.exists()


def test_missing_input_or_revision_is_refused(tmp_path: Path) -> None:
    """Chybějící tabulka i nečitelná revize krok odmítnou."""
    discharge, diagnoses = write_tables(tmp_path)
    candidates, tables = _configs(tmp_path)
    arguments = {
        "diagnoses": diagnoses,
        "candidates": candidates,
        "tables": tables,
        "cohort": tmp_path / "cohort.parquet",
        "manifest": tmp_path / "manifest.json",
    }

    no_table = cohort.run(
        discharge=tmp_path / "none.parquet", revision=_revision, **arguments
    )
    no_git = cohort.run(discharge=discharge, revision=_no_git, **arguments)

    assert no_table is ExitCode.REFUSED
    assert no_git is ExitCode.REFUSED
    assert not (tmp_path / "manifest.json").exists()


def test_config_without_source_table_is_refused(tmp_path: Path) -> None:
    """Config tabulek bez zdrojové tabulky krok odmítne dřív, než zapíše kohortu či manifest."""
    discharge, diagnoses = write_tables(tmp_path)
    candidates, tables = _configs(tmp_path)
    kept = json.loads(tables.read_text(encoding="utf-8"))
    kept["mimic_tables"] = kept["mimic_tables"][:1]
    tables.write_text(json.dumps(kept), encoding="utf-8")

    code = cohort.run(
        discharge=discharge,
        diagnoses=diagnoses,
        candidates=candidates,
        tables=tables,
        cohort=tmp_path / "cohort.parquet",
        manifest=tmp_path / "manifest.json",
        revision=_revision,
    )

    assert code is ExitCode.REFUSED
    assert not (tmp_path / "manifest.json").exists()
