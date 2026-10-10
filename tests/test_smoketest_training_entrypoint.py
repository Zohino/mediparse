"""Vstupní bod kroku train_smoketest_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import RunManifest, SourceRevision
from mediparse.entrypoints import smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import ESTIMATOR, SkopsModelFile
from tests.support import separable_notes

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from mediparse.domain.evaluation import Prediction

PATIENTS: Final = 12
REVISION: Final = SourceRevision(commit="a" * 40, dirty=False)
OVERLAPPING: Final = (
    "diabetes insulin glucose clinic review",
    "fracture cast splint clinic review",
)


def _revision() -> SourceRevision:
    return REVISION


def _no_git() -> SourceRevision:
    msg = "git chybí"
    raise InvalidInputError(msg)


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
    manifest = tmp_path / "build" / "run_manifest.json"

    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=_revision,
    )

    rows = ParquetPredictionTable(predictions).read()
    assert code is ExitCode.OK
    assert rows
    assert {(row.row_id, row.fold) for row in rows} == {("row", 0)}
    assert SkopsModelFile(model).load().classify(["diabetes insulin"])[0].predicted
    written = RunManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    assert written.dataset_sha256 == file_sha256(notes)
    assert written.config_sha256 == file_sha256(config)
    assert written.seed == 0
    assert written.row_id == "row"
    assert written.source == REVISION
    assert written.model_revision is None
    assert written.tokenizer_revision is None
    assert written.model == ESTIMATOR
    assert tuple(written.packages) == smoketest_training.PACKAGES


def test_missing_inputs_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící tabulka i config krok odmítnou a nic nevznikne."""
    notes, config = _inputs(tmp_path)
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"
    manifest = tmp_path / "run_manifest.json"

    no_table = smoketest_training.run(
        notes=tmp_path / "none.parquet",
        config=config,
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=_revision,
    )
    no_config = smoketest_training.run(
        notes=notes,
        config=tmp_path / "none.json",
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=_revision,
    )

    assert no_table is ExitCode.REFUSED
    assert no_config is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()
    assert not manifest.exists()
    assert "none.parquet" in caplog.text
    assert "none.json" in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, diagnosis="aki")
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"
    manifest = tmp_path / "run_manifest.json"

    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=_revision,
    )

    assert code is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()
    assert not manifest.exists()
    assert "aki" in caplog.text


def test_invalid_config_is_refused(tmp_path: Path) -> None:
    """Config bez row_id krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, row_id=None)
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"
    manifest = tmp_path / "run_manifest.json"

    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=_revision,
    )

    assert code is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()
    assert not manifest.exists()


def _predictions(root: Path, seed: int, name: str) -> tuple[Prediction, ...]:
    notes, config = _inputs(root, seed=seed)
    path = root / name
    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=root / f"{name}.skops",
        predictions=path,
        manifest=root / f"{name}.json",
        revision=_revision,
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


def _manifest_bytes(root: Path, seed: int, name: str) -> bytes:
    notes, config = _inputs(root, seed=seed)
    manifest = root / f"{name}.json"
    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=root / f"{name}.skops",
        predictions=root / f"{name}.parquet",
        manifest=manifest,
        revision=_revision,
    )
    assert code is ExitCode.OK
    return manifest.read_bytes()


def test_manifest_is_byte_identical_for_same_inputs(tmp_path: Path) -> None:
    """Dva běhy se stejnými vstupy dají bajtově shodný manifest."""
    assert _manifest_bytes(tmp_path, 0, "first") == _manifest_bytes(
        tmp_path, 0, "second"
    )


def test_seed_changes_manifest(tmp_path: Path) -> None:
    """Jiný seed dá jiný seed i hash configu."""
    first = RunManifest.model_validate_json(_manifest_bytes(tmp_path, 0, "a"))
    other = RunManifest.model_validate_json(_manifest_bytes(tmp_path, 1, "b"))

    assert (first.seed, other.seed) == (0, 1)
    assert first.config_sha256 != other.config_sha256


def test_unreadable_revision_is_refused_before_training(tmp_path: Path) -> None:
    """Revize, kterou nelze přečíst, krok odmítne a nevznikne nic."""
    notes, config = _inputs(tmp_path)
    model = tmp_path / "model.skops"
    predictions = tmp_path / "predictions.parquet"
    manifest = tmp_path / "run_manifest.json"

    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=model,
        predictions=predictions,
        manifest=manifest,
        revision=_no_git,
    )

    assert code is ExitCode.REFUSED
    assert not model.exists()
    assert not predictions.exists()
    assert not manifest.exists()
