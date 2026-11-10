"""Nástroj screeningu: skutečné adaptéry, provenance, stráž dat a determinismus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import ScreeningManifest
from mediparse.entrypoints import classical_screening
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.screening_support import DEVELOPMENT, SHARED, notes
from tests.support import (
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
    REVISION,
    fixed_revision,
    mark_synthetic,
    missing_revision,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from mediparse.domain.environment import SourceRevision

VISITS_PER_PATIENT = 16


@dataclass(frozen=True)
class _Inputs:
    cohort: Path
    split: Path
    config: Path


def _inputs(
    root: Path,
    *,
    preprocessing: str = "none",
    strip: bool = False,
    families: tuple[str, ...] = ("complement_nb", "linear_svm"),
    visits: int = 2,
) -> _Inputs:
    cohort = root / "cohort.parquet"
    ParquetNoteTable(cohort).write(notes(visits=visits))
    split = root / "split.parquet"
    ParquetSplitTable(split).write(SHARED)
    config = root / "config.json"
    config.write_text(
        json.dumps({
            "seed": 7,
            "folds": 2,
            "threshold_folds": 2,
            "lsa_components": 2,
            "input": {
                "strip_diagnostic_sections": strip,
                "text_preprocessing": preprocessing,
            },
            "vectorizers": [
                {"name": "word", "analyzer": "word", "ngram_range": [1, 1]}
            ],
            "families": families,
        }),
        encoding="utf-8",
    )
    return _Inputs(cohort, split, config)


def _run(
    inputs: _Inputs,
    root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    revision: Callable[[], SourceRevision] = fixed_revision,
) -> ExitCode:
    return classical_screening.run(
        environ={} if environ is None else environ,
        cohort=inputs.cohort,
        split=inputs.split,
        config=inputs.config,
        candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
        lexicon=REPOSITORY_SECTION_HEADERS,
        out=root / "classical.parquet",
        manifest=root / "classical_manifest.json",
        revision=revision,
    )


def test_results_manifest_records_actual_inputs_and_development(tmp_path: Path) -> None:
    """Manifest váže agregáty na skutečné soubory, původní split, prostředí a velikost vývoje."""
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    mark_synthetic(tmp_path)
    assert _run(inputs, out, environ={"CODEX_THREAD_ID": "test"}) is ExitCode.OK
    written = ScreeningManifest.model_validate_json(
        (out / "classical_manifest.json").read_text(encoding="utf-8")
    )
    result = out / "classical.parquet"
    assert (
        written.dataset_sha256 == hashlib.sha256(inputs.cohort.read_bytes()).hexdigest()
    )
    assert written.results_sha256 == hashlib.sha256(result.read_bytes()).hexdigest()
    assert (
        written.config_sha256 == hashlib.sha256(inputs.config.read_bytes()).hexdigest()
    )
    shared = ParquetSplitTable(inputs.split).read()
    assert written.split_sha256 == shared.split.sha256()
    assert written.split_seed == shared.config.seed
    assert written.split_folds == shared.config.folds
    development = [note for note in notes() if note.subject_id in DEVELOPMENT]
    assert written.development_notes == len(development)
    assert written.development_patients == len({
        note.subject_id for note in development
    })
    assert written.seed == 7  # ruff: ignore[magic-value-comparison]
    assert written.diagnostic_strip is None
    assert written.text_preprocessing is TextPreprocessing.NONE
    assert written.text_normalization is None
    assert written.environment.source == REVISION
    assert set(written.environment.packages) == {
        "python",
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "pyarrow",
        "pydantic",
    }
    assert all(written.environment.packages.values())
    grid = json.loads(inputs.config.read_text(encoding="utf-8"))
    assert (
        pq.read_table(result).num_rows
        == grid["folds"] * len(grid["vectorizers"]) * len(grid["families"]) * 5
    )


def test_repeated_run_has_identical_result_bytes(tmp_path: Path) -> None:
    """Dva běhy nad stejnými vstupy zapíší bajtově shodné výsledky."""
    inputs = _inputs(
        tmp_path,
        families=("complement_nb", "logreg", "linear_svm", "nb_svm", "lightgbm_lsa"),
        visits=VISITS_PER_PATIENT,
    )
    first, second = tmp_path / "first", tmp_path / "second"
    assert _run(inputs, first) is ExitCode.OK
    assert _run(inputs, second) is ExitCode.OK
    assert (first / "classical.parquet").read_bytes() == (
        second / "classical.parquet"
    ).read_bytes()


@pytest.mark.parametrize("unmarked", ["cohort", "split"])
def test_agent_refuses_each_unmarked_input_before_reading(
    tmp_path: Path, unmarked: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Stráž odmítne každý neoznačený adresář před čtením souborů a nic nezapíše."""
    marked = tmp_path / "marked"
    mark_synthetic(marked)
    paths = {
        name: (tmp_path / "unmarked" if name == unmarked else marked)
        / f"{name}.parquet"
        for name in ("cohort", "split")
    }
    inputs = _Inputs(paths["cohort"], paths["split"], tmp_path / "missing.json")
    out = tmp_path / "out"
    assert _run(inputs, out, environ={"CODEX_THREAD_ID": "test"}) is ExitCode.REFUSED
    assert "Agent Codex nesmí číst data MIMIC" in caplog.text
    assert not out.exists()


@pytest.mark.parametrize("preprocessing", ["normalized", "dictionary"])
def test_manifest_preserves_preprocessing_and_strip(
    tmp_path: Path, preprocessing: str
) -> None:
    """Manifest nese použitý strip, slovník a explicitní úroveň předzpracování."""
    inputs = _inputs(tmp_path, preprocessing=preprocessing, strip=True)
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.OK
    manifest = ScreeningManifest.model_validate_json(
        (out / "classical_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.text_preprocessing.value == preprocessing
    assert manifest.text_normalization is not None
    assert (
        manifest.text_normalization.acronyms["COPD"]
        == "chronic obstructive pulmonary disease"
    )
    assert bool(manifest.text_normalization.variants) == (preprocessing == "dictionary")
    assert manifest.diagnostic_strip is not None
    assert (
        manifest.diagnostic_strip.lexicon_sha256
        == hashlib.sha256(REPOSITORY_SECTION_HEADERS.read_bytes()).hexdigest()
    )


def test_invalid_revision_refuses_before_any_output(tmp_path: Path) -> None:
    """Nečitelná provenance zabrání výpočtu i zápisu výsledků."""
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(inputs, out, revision=missing_revision) is ExitCode.REFUSED
    assert not out.exists()


def test_main_parses_required_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI předá všechny explicitní cesty a návratový kód run."""
    names = ("cohort", "split", "config", "candidates", "lexicon", "out", "manifest")
    paths = {name: tmp_path / name for name in names}
    monkeypatch.setattr(
        "sys.argv",
        [
            "mediparse-screening-classical",
            *(
                argument
                for name, path in paths.items()
                for argument in (f"--{name}", str(path))
            ),
        ],
    )
    execute = Mock(return_value=ExitCode.REFUSED)
    monkeypatch.setattr(classical_screening, "run", execute)
    monkeypatch.setattr(classical_screening, "configure_logging", Mock())
    assert classical_screening.main() is ExitCode.REFUSED
    assert {name: execute.call_args.kwargs[name] for name in names} == paths


def test_preprocessing_changes_screening_result_bytes(tmp_path: Path) -> None:
    """Zkratka COPD změní příznaky a agregované výsledky při dictionary oproti none."""
    corpus = tuple(
        replace(
            note,
            text=(
                "COPD wheeze"
                if note.has_label(Diagnosis.DIABETES)
                else "chronic obstructive pulmonary disease cough"
            ),
        )
        for note in notes()
    )
    results = []
    for preprocessing in ("none", "dictionary"):
        root = tmp_path / preprocessing
        inputs = _inputs(root, preprocessing=preprocessing)
        ParquetNoteTable(inputs.cohort).write(corpus)
        out = root / "out"
        assert _run(inputs, out) is ExitCode.OK
        results.append((out / "classical.parquet").read_bytes())

    assert results[0] != results[1]
