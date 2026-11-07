"""Vstupní bod kroku build_cohort: kohorta a manifest, odmítnutí malé buňky."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Final

import pyarrow.parquet as pq
import pytest

from mediparse.domain.cohort import MIN_CELL, CohortManifest, FunnelStepName
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.mimic_table import TableProvenance
from mediparse.entrypoints import cohort
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.table_provenance import (
    provenance_path,
    write_table_provenance,
)
from tests.cohort_support import (
    EXPECTED_FUNNEL,
    EXPECTED_INVALID_CODES,
    EXPECTED_POSITIVES,
    icd,
    note,
    text,
    write_candidates,
    write_tables,
)
from tests.support import REVISION, fixed_revision, missing_revision

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mediparse.domain.environment import SourceRevision
    from tests.cohort_support import IcdRow, NoteRow

DISCHARGE_SHA: Final = hashlib.sha256(b"synthetic discharge CSV").hexdigest()
DIAGNOSES_SHA: Final = hashlib.sha256(b"synthetic diagnoses CSV").hexdigest()


def _tables(
    root: Path,
    notes: list[NoteRow] | None = None,
    codes: list[IcdRow] | None = None,
    *,
    padded: bool = True,
) -> tuple[Path, Path]:
    discharge, diagnoses = write_tables(root, notes, codes, padded=padded)
    for path, digest in ((discharge, DISCHARGE_SHA), (diagnoses, DIAGNOSES_SHA)):
        write_table_provenance(path, source=f"{path.stem}.csv.gz", source_sha256=digest)
    return discharge, diagnoses


def _configs(root: Path) -> tuple[Path, Path]:
    candidates = write_candidates(root / "diagnosis_candidates.json")
    rule = root / "input_corpus.json"
    rule.write_text('{"min_chars":100}', encoding="utf-8")
    return candidates, rule


def _cohort(
    tmp_path: Path,
    discharge: Path,
    diagnoses: Path,
    *,
    configs: tuple[Path, Path],
    revision: Callable[[], SourceRevision] = fixed_revision,
) -> ExitCode:
    candidates, corpus_rule = configs
    return cohort.run(
        discharge=discharge,
        diagnoses=diagnoses,
        candidates=candidates,
        corpus_rule=corpus_rule,
        cohort=tmp_path / "resources" / "cohort.parquet",
        manifest=tmp_path / "results" / "cohort" / "manifest.json",
        revision=revision,
    )


def test_writes_cohort_and_manifest(tmp_path: Path) -> None:
    """Kohorta a manifest vzniknou; otisky odpovídají skutečně čteným parquetům a ověřeným CSV."""
    discharge, diagnoses = _tables(tmp_path)
    candidates, rule = _configs(tmp_path)
    out = tmp_path / "resources" / "cohort.parquet"
    manifest = tmp_path / "results" / "cohort" / "manifest.json"

    code = _cohort(
        tmp_path,
        discharge,
        diagnoses,
        configs=(candidates, rule),
        revision=fixed_revision,
    )

    written = CohortManifest.model_validate_json(
        manifest.read_text(encoding="utf-8"), strict=False
    )
    assert code is ExitCode.OK
    assert written.cohort_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert written.config_sha256 == hashlib.sha256(candidates.read_bytes()).hexdigest()
    assert written.input_corpus_sha256 == hashlib.sha256(rule.read_bytes()).hexdigest()
    assert written.sources == {
        "discharge": TableProvenance(
            source="discharge.csv.gz",
            source_sha256=DISCHARGE_SHA,
            sha256=hashlib.sha256(discharge.read_bytes()).hexdigest(),
        ),
        "diagnoses_icd": TableProvenance(
            source="diagnoses_icd.csv.gz",
            source_sha256=DIAGNOSES_SHA,
            sha256=hashlib.sha256(diagnoses.read_bytes()).hexdigest(),
        ),
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
    discharge, diagnoses = _tables(
        tmp_path,
        [note("1-DS-1", 1, 10, text("a"))],
        [icd(10, 1, "E119", "10")],
        padded=False,
    )
    candidates, rule = _configs(tmp_path)
    out = tmp_path / "resources" / "cohort.parquet"
    manifest = tmp_path / "results" / "cohort" / "manifest.json"

    code = _cohort(
        tmp_path,
        discharge,
        diagnoses,
        configs=(candidates, rule),
        revision=fixed_revision,
    )

    assert code is ExitCode.REFUSED
    assert any("diabetes" in message for message in caplog.messages)
    numbers = {n for message in caplog.messages for n in re.findall(r"\d+", message)}
    assert numbers <= {str(MIN_CELL)}
    assert not out.exists()
    assert not manifest.exists()


def test_missing_input_table_is_refused(tmp_path: Path) -> None:
    """Chybějící vstupní tabulka krok odmítne, kohorta ani manifest nevzniknou."""
    _, diagnoses = _tables(tmp_path)
    candidates, rule = _configs(tmp_path)
    code = _cohort(
        tmp_path,
        tmp_path / "none.parquet",
        diagnoses,
        configs=(candidates, rule),
    )

    assert code is ExitCode.REFUSED
    assert not (tmp_path / "resources" / "cohort.parquet").exists()
    assert not (tmp_path / "results" / "cohort" / "manifest.json").exists()


def test_unreadable_revision_is_refused(tmp_path: Path) -> None:
    """Nečitelná revize repa krok odmítne, kohorta ani manifest nevzniknou."""
    discharge, diagnoses = _tables(tmp_path)
    candidates, rule = _configs(tmp_path)
    code = _cohort(
        tmp_path,
        discharge,
        diagnoses,
        configs=(candidates, rule),
        revision=missing_revision,
    )

    assert code is ExitCode.REFUSED
    assert not (tmp_path / "resources" / "cohort.parquet").exists()
    assert not (tmp_path / "results" / "cohort" / "manifest.json").exists()


@pytest.mark.parametrize("table", ["discharge", "diagnoses_icd"])
def test_changed_parquet_is_refused_before_writing(tmp_path: Path, table: str) -> None:
    """Platný parquet změněný po zápisu vazby nesmí vytvořit kohortu ani manifest."""
    discharge, diagnoses = _tables(tmp_path)
    candidates, rule = _configs(tmp_path)
    changed = tmp_path / f"{table}.parquet"
    rows = pq.read_table(changed).replace_schema_metadata({b"changed": b"yes"})
    pq.write_table(rows, changed)

    code = _cohort(
        tmp_path,
        discharge,
        diagnoses,
        configs=(candidates, rule),
    )

    assert code is ExitCode.REFUSED
    assert not (tmp_path / "resources" / "cohort.parquet").exists()
    assert not (tmp_path / "results" / "cohort" / "manifest.json").exists()


def test_missing_source_is_refused_before_writing(tmp_path: Path) -> None:
    """Kohorta bez vazby jedné tabulky nevznikne."""
    discharge, diagnoses = _tables(tmp_path)
    candidates, rule = _configs(tmp_path)
    provenance_path(diagnoses).unlink()

    assert (
        _cohort(
            tmp_path,
            discharge,
            diagnoses,
            configs=(candidates, rule),
        )
        is ExitCode.REFUSED
    )
    assert not (tmp_path / "resources" / "cohort.parquet").exists()
    assert not (tmp_path / "results" / "cohort" / "manifest.json").exists()


def test_rule_changes_manifest_hash_and_funnel(tmp_path: Path) -> None:
    """Práh tří znaků pustí krátkou zprávu ze základu a má vlastní otisk configu."""
    discharge, diagnoses = _tables(tmp_path)
    candidates, _ = _configs(tmp_path)
    rule = tmp_path / "custom_rule.json"
    manifest = tmp_path / "results" / "cohort" / "manifest.json"
    counts = []
    hashes = []
    for threshold in (100, 3):
        rule.write_text(json.dumps({"min_chars": threshold}), encoding="utf-8")
        assert (
            _cohort(
                tmp_path,
                discharge,
                diagnoses,
                configs=(candidates, rule),
            )
            is ExitCode.OK
        )
        written = CohortManifest.model_validate_json(
            manifest.read_text(encoding="utf-8"), strict=False
        )
        assert (
            written.input_corpus_sha256 == hashlib.sha256(rule.read_bytes()).hexdigest()
        )
        assert (
            written.config_sha256 == hashlib.sha256(candidates.read_bytes()).hexdigest()
        )
        counts.append({
            step.step: (step.notes, step.patients) for step in written.counts.funnel
        })
        hashes.append(written.input_corpus_sha256)

    # Padding přidá 50 zpráv od 50 pacientů; práh 3 navíc propustí zprávu 5-DS-1.
    short_funnel = EXPECTED_FUNNEL | {
        step: (EXPECTED_FUNNEL[step][0] + 1, EXPECTED_FUNNEL[step][1] + 1)
        for step in (FunnelStepName.LONG_ENOUGH, FunnelStepName.UNIQUE)
    }
    assert counts == [EXPECTED_FUNNEL, short_funnel]
    assert hashes[0] != hashes[1]
