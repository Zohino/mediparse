"""Vstupní bod kroku train_smoketest_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from mediparse.entrypoints import smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import SkopsModelFile
from tests.support import separable_notes

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from mediparse.domain.evaluation import Prediction

PATIENTS: Final = 12
OVERLAPPING: Final = (
    "diabetes insulin glucose clinic review",
    "fracture cast splint clinic review",
)


def _inputs(
    root: Path,
    *,
    diagnosis: str = "diabetes",
    seed: int = 0,
    row_id: str | None = "row",
) -> tuple[Path, Path]:
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(separable_notes(PATIENTS, OVERLAPPING))
    config = root / f"config-{seed}.json"
    config.write_text(
        json.dumps({
            **({} if row_id is None else {"row_id": row_id}),
            "diagnosis": diagnosis,
            "seed": seed,
            "folds": 2,
            "regularization": 1.0,
        }),
        encoding="utf-8",
    )
    return notes, config


def test_writes_model_and_predictions(tmp_path: Path) -> None:
    """Běh uloží načitatelný model a predikce testovacích zpráv s row_id a foldem 0."""
    notes, config = _inputs(tmp_path)
    model = tmp_path / "build" / "model.skops"
    predictions = tmp_path / "build" / "predictions.parquet"

    code = smoketest_training.run(
        notes=notes, config=config, model=model, predictions=predictions
    )

    rows = ParquetPredictionTable(predictions).read()
    assert code is ExitCode.OK
    assert rows
    assert {(row.row_id, row.fold) for row in rows} == {("row", 0)}
    assert SkopsModelFile(model).load().classify(["diabetes insulin"])[0].predicted


def test_missing_inputs_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící tabulka i config krok odmítnou a nic nevznikne."""
    notes, config = _inputs(tmp_path)
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"

    no_table = smoketest_training.run(
        notes=tmp_path / "none.parquet",
        config=config,
        model=model,
        predictions=predictions,
    )
    no_config = smoketest_training.run(
        notes=notes, config=tmp_path / "none.json", model=model, predictions=predictions
    )

    assert no_table is ExitCode.REFUSED
    assert no_config is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()
    assert "none.parquet" in caplog.text
    assert "none.json" in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, diagnosis="aki")
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"

    code = smoketest_training.run(
        notes=notes, config=config, model=model, predictions=predictions
    )

    assert code is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()
    assert "aki" in caplog.text


def test_invalid_config_is_refused(tmp_path: Path) -> None:
    """Config bez row_id krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, row_id=None)
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"

    code = smoketest_training.run(
        notes=notes, config=config, model=model, predictions=predictions
    )

    assert code is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()


def _predictions(root: Path, seed: int, name: str) -> tuple[Prediction, ...]:
    notes, config = _inputs(root, seed=seed)
    path = root / name
    code = smoketest_training.run(
        notes=notes, config=config, model=root / f"{name}.skops", predictions=path
    )
    assert code is ExitCode.OK
    return ParquetPredictionTable(path).read()


def test_seed_decides_the_split(tmp_path: Path) -> None:
    """Stejný seed dá shodný obsah predikcí, jiný seed jiné testovací zprávy."""
    first = _predictions(tmp_path, 0, "first.parquet")
    repeated = _predictions(tmp_path, 0, "repeated.parquet")
    others = [
        _predictions(tmp_path, seed, f"seed{seed}.parquet") for seed in range(1, 6)
    ]

    assert first == repeated
    assert len({tuple(row.note_id for row in rows) for rows in [first, *others]}) > 1
