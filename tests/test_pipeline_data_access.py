"""Stráž veřejných kroků a nativní cesta syntetiky v prostředí Codexu."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.entrypoints import (
    cohort,
    mimic_parquet,
    model_evaluation,
    model_inference,
    model_training,
    split_materialization,
)
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.table_provenance import write_table_provenance
from tests.support import (
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_INPUT_CORPUS,
    REPOSITORY_SECTION_HEADERS,
    REPOSITORY_SPLIT_CONFIG,
    REPOSITORY_TRAINING_CONFIG,
    fixed_revision,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

BLOCKED: Final = {"CODEX_THREAD_ID": "x"}
INPUTS: Final = {
    "mimic_parquet": ("mimic/parquet/table.csv.gz",),
    "cohort": (
        "mimic/parquet/discharge.parquet",
        "mimic/parquet/diagnoses_icd.parquet",
    ),
    "split_materialization": ("cohort.parquet",),
    "model_training": ("cohort.parquet", "splits.parquet"),
    "model_evaluation": ("predictions.parquet", "cohort.parquet", "splits.parquet"),
    "model_inference": ("cohort.parquet", "model.skops", "predictions.parquet"),
}


def _steps(root: Path, environ: Mapping[str, str]) -> dict[str, Callable[[], ExitCode]]:
    tables = root / "mimic/parquet"
    notes = root / "cohort.parquet"
    split = root / "splits.parquet"
    model = root / "model.skops"
    predictions = root / "predictions.parquet"
    return {
        "mimic_parquet": partial(
            mimic_parquet.run,
            table=tables / "table.csv.gz",
            inventory=root / "inventory.json",
            parquet=root / "converted.parquet",
            environ=environ,
        ),
        "cohort": partial(
            cohort.run,
            discharge=tables / "discharge.parquet",
            diagnoses=tables / "diagnoses_icd.parquet",
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            corpus_rule=REPOSITORY_INPUT_CORPUS,
            cohort=notes,
            manifest=root / "cohort/manifest.json",
            revision=fixed_revision,
            environ=environ,
        ),
        "split_materialization": partial(
            split_materialization.run,
            cohort=notes,
            config=REPOSITORY_SPLIT_CONFIG,
            output=split,
            environ=environ,
        ),
        "model_training": partial(
            model_training.run,
            cohort=notes,
            config=REPOSITORY_TRAINING_CONFIG,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            lexicon=REPOSITORY_SECTION_HEADERS,
            split=split,
            model=model,
            predictions=predictions,
            manifest=root / "run_manifest.json",
            costs=root / "training_costs.json",
            revision=fixed_revision,
            cuda_visible_devices="",
            cuda_device_order=None,
            environ=environ,
        ),
        "model_evaluation": partial(
            model_evaluation.run,
            predictions=predictions,
            cohort=notes,
            split=split,
            metrics=root / "metrics.parquet",
            environ=environ,
        ),
        "model_inference": partial(
            model_inference.run,
            cohort=notes,
            config=REPOSITORY_TRAINING_CONFIG,
            lexicon=REPOSITORY_SECTION_HEADERS,
            model=model,
            predictions=predictions,
            inference=root / "inference.parquet",
            costs=root / "inference_costs.json",
            cuda_visible_devices="",
            cuda_device_order=None,
            environ=environ,
        ),
    }


@pytest.mark.parametrize("entrypoint", INPUTS)
def test_entrypoint_refuses_before_opening_input(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, entrypoint: str
) -> None:
    """Stráž má přednost před rozbitým vstupem a nevytvoří žádný výstup."""
    for relative in INPUTS[entrypoint]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not parquet")
        if entrypoint == "cohort":
            write_table_provenance(
                path,
                source=f"https://mediparse.invalid/synthetic/{path.stem}",
                source_sha256=hashlib.sha256(b"not parquet").hexdigest(),
            )
    if entrypoint == "mimic_parquet":
        (tmp_path / "inventory.json").write_text(
            json.dumps({
                "file": "table.csv.gz",
                "sha256": hashlib.sha256(b"not parquet").hexdigest(),
                "bytes": len(b"not parquet"),
                "records": 1,
                "columns": ["a"],
            }),
            encoding="utf-8",
        )
    before = set(tmp_path.rglob("*"))

    assert _steps(tmp_path, BLOCKED)[entrypoint]() is ExitCode.REFUSED

    assert "Agent Codex nesmí číst data MIMIC" in caplog.text
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.usefixtures("synthetic_tables")
def test_native_synthetic_pipeline_under_codex(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Značky dovolí celou cestu smoketestu; odebrání značky artefaktů zastaví split."""
    steps = _steps(tmp_path, BLOCKED)
    for name in INPUTS:
        if name != "mimic_parquet":
            assert steps[name]() is ExitCode.OK, name

    (tmp_path / "synthetic.json").unlink()
    assert steps["split_materialization"]() is ExitCode.REFUSED
    assert "Agent Codex nesmí číst data MIMIC" in caplog.text
