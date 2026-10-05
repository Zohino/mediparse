"""Vstupní bod kroku train_smoketest_model: model a metriky, odmítnutí a determinismus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from mediparse.entrypoints import smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.sklearn_classifier import SkopsModelFile
from tests.support import separable_notes

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

PATIENTS: Final = 12
OVERLAPPING: Final = (
    "diabetes insulin glucose clinic review",
    "fracture cast splint clinic review",
)
KEYS: Final = {
    "accuracy",
    "diagnosis",
    "f1",
    "precision",
    "recall",
    "test_notes",
    "test_positives",
    "train_notes",
}


def _inputs(
    root: Path, *, diagnosis: str = "diabetes", seed: int = 0
) -> tuple[Path, Path]:
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(separable_notes(PATIENTS, OVERLAPPING))
    config = root / f"config-{seed}.json"
    config.write_text(
        json.dumps({
            "diagnosis": diagnosis,
            "seed": seed,
            "folds": 2,
            "regularization": 1.0,
        }),
        encoding="utf-8",
    )
    return notes, config


def test_writes_model_and_metrics(tmp_path: Path) -> None:
    """Běh uloží načitatelný model a metrics.json s očekávanými klíči."""
    notes, config = _inputs(tmp_path)
    model = tmp_path / "build" / "model.skops"
    metrics = tmp_path / "build" / "metrics.json"

    code = smoketest_training.run(
        notes=notes, config=config, model=model, metrics=metrics
    )

    assert code is ExitCode.OK
    assert set(json.loads(metrics.read_text(encoding="utf-8"))) == KEYS
    assert SkopsModelFile(model).load().predict(["diabetes insulin"]) == (True,)


def test_missing_inputs_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící tabulka i config krok odmítnou a nic nevznikne."""
    notes, config = _inputs(tmp_path)
    model = tmp_path / "model.skops"
    metrics = tmp_path / "metrics.json"

    no_table = smoketest_training.run(
        notes=tmp_path / "none.parquet", config=config, model=model, metrics=metrics
    )
    no_config = smoketest_training.run(
        notes=notes, config=tmp_path / "none.json", model=model, metrics=metrics
    )

    assert no_table is ExitCode.REFUSED
    assert no_config is ExitCode.REFUSED
    assert not model.exists()
    assert not metrics.exists()
    assert "none.parquet" in caplog.text
    assert "none.json" in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, diagnosis="aki")
    model = tmp_path / "model.skops"
    metrics = tmp_path / "metrics.json"

    code = smoketest_training.run(
        notes=notes, config=config, model=model, metrics=metrics
    )

    assert code is ExitCode.REFUSED
    assert not model.exists()
    assert not metrics.exists()
    assert "aki" in caplog.text


def _metrics(root: Path, seed: int, name: str) -> Path:
    notes, config = _inputs(root, seed=seed)
    metrics = root / name
    code = smoketest_training.run(
        notes=notes, config=config, model=root / f"{name}.skops", metrics=metrics
    )
    assert code is ExitCode.OK
    return metrics


def test_seed_decides_the_split(tmp_path: Path) -> None:
    """Stejný seed dá shodné bajty metrics.json, jiný seed jiné velikosti částí."""
    first = _metrics(tmp_path, 0, "first.json")
    repeated = _metrics(tmp_path, 0, "repeated.json")
    others = [_metrics(tmp_path, seed, f"seed{seed}.json") for seed in range(1, 6)]

    assert first.read_bytes() == repeated.read_bytes()
    sizes = {
        json.loads(path.read_text(encoding="utf-8"))["train_notes"]
        for path in [first, *others]
    }
    assert len(sizes) > 1
