"""Vstupní bod kroku build_cohort: kohorta a manifest, odmítnutí malé buňky."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Final

from mediparse.domain.cohort import MIN_CELL, CohortManifest, FunnelStepName
from mediparse.domain.diagnosis import Diagnosis
from mediparse.entrypoints import cohort
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from tests.cohort_support import (
    EXPECTED_FUNNEL,
    EXPECTED_INVALID_CODES,
    EXPECTED_POSITIVES,
    candidate_prefixes,
    icd,
    note,
    text,
    write_tables,
)
from tests.support import REVISION, fixed_revision, missing_revision

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from mediparse.domain.environment import SourceRevision

DISCHARGE_SHA: Final = "1" * 64
DIAGNOSES_SHA: Final = "2" * 64


def _configs(root: Path) -> tuple[Path, Path]:
    candidates = root / "diagnosis_candidates.json"
    candidates.write_text(
        json.dumps({"candidates": candidate_prefixes()}),
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
        revision=fixed_revision,
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
    assert written.invalid_codes == EXPECTED_INVALID_CODES
    assert written.counts.funnel[-1].notes == EXPECTED_FUNNEL[FunnelStepName.UNIQUE][0]
    assert (
        written.counts.positives[Diagnosis.COPD] == EXPECTED_POSITIVES[Diagnosis.COPD]
    )
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
        revision=fixed_revision,
    )

    assert code is ExitCode.REFUSED
    assert any("diabetes" in message for message in caplog.messages)
    numbers = {n for message in caplog.messages for n in re.findall(r"\d+", message)}
    assert numbers <= {str(MIN_CELL)}
    assert not out.exists()
    assert not manifest.exists()


def _run(
    tmp_path: Path, *, discharge: Path | None, revision: Callable[[], SourceRevision]
) -> ExitCode:
    existing, diagnoses = write_tables(tmp_path)
    candidates, tables = _configs(tmp_path)
    return cohort.run(
        discharge=discharge or existing,
        diagnoses=diagnoses,
        candidates=candidates,
        tables=tables,
        cohort=tmp_path / "cohort.parquet",
        manifest=tmp_path / "manifest.json",
        revision=revision,
    )


def test_missing_input_table_is_refused(tmp_path: Path) -> None:
    """Chybějící vstupní tabulka krok odmítne, kohorta ani manifest nevzniknou."""
    code = _run(tmp_path, discharge=tmp_path / "none.parquet", revision=fixed_revision)

    assert code is ExitCode.REFUSED
    assert not (tmp_path / "cohort.parquet").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_unreadable_revision_is_refused(tmp_path: Path) -> None:
    """Nečitelná revize repa krok odmítne, kohorta ani manifest nevzniknou."""
    code = _run(tmp_path, discharge=None, revision=missing_revision)

    assert code is ExitCode.REFUSED
    assert not (tmp_path / "cohort.parquet").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_config_without_source_table_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
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
        revision=fixed_revision,
    )

    assert code is ExitCode.REFUSED
    assert any("diagnoses_icd.csv.gz" in message for message in caplog.messages)
    assert not (tmp_path / "cohort.parquet").exists()
    assert not (tmp_path / "manifest.json").exists()
