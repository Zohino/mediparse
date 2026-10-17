"""Vstupní bod kroku train_smoketest_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    from collections.abc import Callable
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


@dataclass(frozen=True)
class _Outputs:
    """Výstupy jednoho běhu kroku v jednom adresáři."""

    model: Path
    predictions: Path
    manifest: Path

    @classmethod
    def under(cls, directory: Path) -> _Outputs:
        return cls(
            directory / "model.skops",
            directory / "predictions.parquet",
            directory / "run_manifest.json",
        )

    def existing(self) -> list[Path]:
        return [
            path
            for path in (self.model, self.predictions, self.manifest)
            if path.exists()
        ]


def _run(
    notes: Path,
    config: Path,
    outputs: _Outputs,
    revision: Callable[[], SourceRevision] = _revision,
) -> ExitCode:
    return smoketest_training.run(
        notes=notes,
        config=config,
        model=outputs.model,
        predictions=outputs.predictions,
        manifest=outputs.manifest,
        revision=revision,
    )


def test_writes_model_and_predictions(tmp_path: Path) -> None:
    """Běh uloží načitatelný model a predikce testovacích zpráv s row_id a foldem 0."""
    notes, config = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "build")

    code = _run(notes, config, outputs)

    rows = ParquetPredictionTable(outputs.predictions).read()
    assert code is ExitCode.OK
    assert rows
    assert {(row.row_id, row.fold) for row in rows} == {("row", 0)}
    assert (
        SkopsModelFile(outputs.model).load().classify(["diabetes insulin"])[0].predicted
    )
    written = RunManifest.model_validate_json(
        outputs.manifest.read_text(encoding="utf-8")
    )
    assert written.dataset_sha256 == file_sha256(notes)
    assert written.config_sha256 == file_sha256(config)
    assert written.seed == 0
    assert written.row_id == "row"
    assert written.environment.source == REVISION
    assert written.model_revision is None
    assert written.tokenizer_revision is None
    assert written.model == ESTIMATOR
    assert tuple(written.environment.packages) == smoketest_training.PACKAGES


def test_missing_inputs_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící tabulka i config krok odmítnou a nic nevznikne."""
    notes, config = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "out")

    no_table = _run(tmp_path / "none.parquet", config, outputs)
    no_config = _run(notes, tmp_path / "none.json", outputs)

    assert no_table is ExitCode.REFUSED
    assert no_config is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "none.parquet" in caplog.text
    assert "none.json" in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, diagnosis="pneumonia")
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "pneumonia" in caplog.text


def test_invalid_config_is_refused(tmp_path: Path) -> None:
    """Config bez row_id krok odmítne a nic nevznikne."""
    notes, config = _inputs(tmp_path, row_id=None)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []


def _successful_run(root: Path, seed: int, name: str) -> _Outputs:
    notes, config = _inputs(root, seed=seed)
    outputs = _Outputs.under(root / name)
    assert _run(notes, config, outputs) is ExitCode.OK
    return outputs


def _predictions(root: Path, seed: int, name: str) -> tuple[Prediction, ...]:
    return ParquetPredictionTable(_successful_run(root, seed, name).predictions).read()


def test_seed_decides_the_split(tmp_path: Path) -> None:
    """Stejný seed dá stejný obsah predikcí, jiný seed jiné testovací zprávy."""
    first = _predictions(tmp_path, 0, "first")
    repeated = _predictions(tmp_path, 0, "repeated")
    others = [_predictions(tmp_path, seed, f"seed{seed}") for seed in range(1, 6)]

    assert first == repeated
    assert len({tuple(row.note_id for row in rows) for rows in [first, *others]}) > 1


def _manifest_bytes(root: Path, seed: int, name: str) -> bytes:
    return _successful_run(root, seed, name).manifest.read_bytes()


def test_manifest_is_byte_identical_for_same_inputs(tmp_path: Path) -> None:
    """Dva běhy se stejnými vstupy dají bajtově totožný manifest."""
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
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, outputs, revision=_no_git)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
