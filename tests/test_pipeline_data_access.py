"""Stráž veřejných kroků a nativní cesta syntetiky v prostředí Codexu."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
    mark_synthetic,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

BLOCKED: Final = {"CODEX_THREAD_ID": "x"}


def _steps(root: Path, environ: Mapping[str, str]) -> dict[str, partial[ExitCode]]:
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


def _prepare_inventory(root: Path, inputs: Mapping[str, Path]) -> None:
    table = inputs["table"]
    (root / "inventory.json").write_text(
        json.dumps({
            "file": table.name,
            "sha256": hashlib.sha256(b"not parquet").hexdigest(),
            "bytes": len(b"not parquet"),
            "records": 1,
            "columns": ["a"],
        }),
        encoding="utf-8",
    )


def _prepare_sources(_root: Path, inputs: Mapping[str, Path]) -> None:
    for path in inputs.values():
        write_table_provenance(
            path,
            source=f"https://mediparse.invalid/synthetic/{path.stem}",
            source_sha256=hashlib.sha256(b"not parquet").hexdigest(),
        )


@dataclass(frozen=True)
class _GuardCase:
    inputs: tuple[str, ...]
    prepare: Callable[[Path, Mapping[str, Path]], None] | None = None


GUARD_CASES: Final = {
    "mimic_parquet": _GuardCase(("table",), _prepare_inventory),
    "cohort": _GuardCase(("discharge", "diagnoses"), _prepare_sources),
    "split_materialization": _GuardCase(("cohort",)),
    "model_training": _GuardCase(("cohort", "split")),
    "model_evaluation": _GuardCase(("predictions", "cohort", "split")),
    "model_inference": _GuardCase(("cohort", "model", "predictions")),
}


@pytest.mark.parametrize(
    ("entrypoint", "unmarked"),
    [(name, item) for name, case in GUARD_CASES.items() for item in case.inputs],
)
def test_entrypoint_refuses_before_opening_input(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    entrypoint: str,
    unmarked: str,
) -> None:
    """Každý datový vstup má vlastní stráž před otevřením a bez nového výstupu."""
    step = _steps(tmp_path, BLOCKED)[entrypoint]
    case = GUARD_CASES[entrypoint]
    inputs = {name: tmp_path / name / step.keywords[name].name for name in case.inputs}
    for name, path in inputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not parquet")
        if name != unmarked:
            mark_synthetic(path.parent)
    if case.prepare is not None:
        case.prepare(tmp_path, inputs)
    before = set(tmp_path.rglob("*"))

    assert step(**inputs) is ExitCode.REFUSED

    assert "Agent Codex nesmí číst data MIMIC" in caplog.text
    assert str(inputs[unmarked].parent) in caplog.text
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.usefixtures("synthetic_tables")
def test_native_synthetic_pipeline_under_codex(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Značky dovolí celou cestu smoketestu; odebrání značky artefaktů zastaví split."""
    steps = _steps(tmp_path, BLOCKED)
    for name in GUARD_CASES:
        if name != "mimic_parquet":
            assert steps[name]() is ExitCode.OK, name

    (tmp_path / "synthetic.json").unlink()
    assert steps["split_materialization"]() is ExitCode.REFUSED
    assert "Agent Codex nesmí číst data MIMIC" in caplog.text
