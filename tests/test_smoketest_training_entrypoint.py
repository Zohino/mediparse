"""Vstupní bod kroku train_smoketest_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.run_manifest import RunManifest
from mediparse.entrypoints import smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import ESTIMATOR, SkopsModelFile
from tests.model_support import separable_notes, split_config_json, training_config
from tests.support import REVISION, fixed_revision, missing_revision

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mediparse.domain.environment import SourceRevision
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
    split_seed: int = 0,
) -> tuple[Path, Path, Path]:
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(separable_notes(PATIENTS, OVERLAPPING))
    config_body = training_config(diagnosis=diagnosis, seed=seed)
    if row_id is None:
        del config_body["row_id"]
    else:
        config_body["row_id"] = row_id
    config = root / f"config-{seed}.json"
    config.write_text(
        json.dumps(config_body),
        encoding="utf-8",
    )
    split = root / f"split-{split_seed}.json"
    split.write_text(json.dumps(split_config_json(seed=split_seed)), encoding="utf-8")
    return notes, config, split


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
    split: Path,
    outputs: _Outputs,
    revision: Callable[[], SourceRevision] = fixed_revision,
) -> ExitCode:
    return smoketest_training.run(
        notes=notes,
        config=config,
        split=split,
        model=outputs.model,
        predictions=outputs.predictions,
        manifest=outputs.manifest,
        revision=revision,
    )


def test_writes_model_and_predictions(tmp_path: Path) -> None:
    """Běh uloží načitatelný model a predikce testovacích zpráv s row_id a foldem 0."""
    notes, config, split = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "build")

    code = _run(notes, config, split, outputs)

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
    assert written.split_seed == 0
    assert written.split_folds == 2  # ruff: ignore[magic-value-comparison]
    assert written.row_id == "row"
    assert written.environment.source == REVISION
    assert written.model_revision is None
    assert written.tokenizer_revision is None
    assert written.model == ESTIMATOR
    assert tuple(written.environment.packages) == smoketest_training.PACKAGES


@pytest.mark.parametrize(
    ("index", "missing"),
    [(0, "none.parquet"), (1, "none.json"), (2, "none-split.json")],
    ids=["table", "config", "split"],
)
def test_missing_input_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, index: int, missing: str
) -> None:
    """Chybějící tabulka, config tréninku nebo config splitu krok odmítne, nic nevznikne a log jmenuje soubor."""
    paths = list(_inputs(tmp_path))
    paths[index] = tmp_path / missing
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(paths[0], paths[1], paths[2], outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert missing in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    notes, config, split = _inputs(tmp_path, diagnosis="pneumonia")
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "pneumonia" in caplog.text


def test_invalid_config_is_refused(tmp_path: Path) -> None:
    """Config bez row_id krok odmítne a nic nevznikne."""
    notes, config, split = _inputs(tmp_path, row_id=None)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []


def test_invalid_split_config_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Config splitu s jedním foldem krok odmítne, nic nevznikne a log jmenuje soubor."""
    notes, config, split = _inputs(tmp_path)
    split.write_text(json.dumps(split_config_json(folds=1)), encoding="utf-8")
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert split.name in caplog.text


def test_split_seed_outside_generator_range_is_refused(tmp_path: Path) -> None:
    """Záporný seed splitu krok odmítne a nic nevznikne, ne traceback z adaptéru."""
    notes, config, split = _inputs(tmp_path)
    split.write_text(json.dumps(split_config_json(seed=-1)), encoding="utf-8")
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []


def _successful_run(root: Path, seed: int, name: str, split_seed: int = 0) -> _Outputs:
    notes, config, split = _inputs(root, seed=seed, split_seed=split_seed)
    outputs = _Outputs.under(root / name)
    assert _run(notes, config, split, outputs) is ExitCode.OK
    return outputs


def _predictions(
    root: Path, seed: int, name: str, split_seed: int = 0
) -> tuple[Prediction, ...]:
    return ParquetPredictionTable(
        _successful_run(root, seed, name, split_seed).predictions
    ).read()


def _manifest(root: Path, seed: int, name: str, split_seed: int = 0) -> RunManifest:
    return RunManifest.model_validate_json(
        _successful_run(root, seed, name, split_seed).manifest.read_text(
            encoding="utf-8"
        )
    )


def _test_patients(rows: tuple[Prediction, ...]) -> frozenset[tuple[int, int]]:
    return frozenset((row.subject_id, row.fold) for row in rows)


def test_model_seed_does_not_change_test_patients(tmp_path: Path) -> None:
    """Změna seedu modelu při stejném configu splitu nechá testovací pacienty beze změny."""
    patients = {
        _test_patients(_predictions(tmp_path, seed, f"seed{seed}")) for seed in range(6)
    }

    assert len(patients) == 1


def test_model_seed_does_not_change_split_fingerprint(tmp_path: Path) -> None:
    """Otisk splitu v manifestu je pro všechny seedy modelu stejný."""
    fingerprints = {
        _manifest(tmp_path, seed, f"seed{seed}").split_sha256 for seed in range(6)
    }

    assert len(fingerprints) == 1


def test_split_seed_changes_test_patients(tmp_path: Path) -> None:
    """Pozitivní kontrola: seed splitu při stejném configu tréninku mění testovací pacienty."""
    patients = {
        _test_patients(_predictions(tmp_path, 0, f"split{seed}", split_seed=seed))
        for seed in range(6)
    }

    assert len(patients) > 1


def test_split_seed_changes_fingerprint_and_manifest_seed(tmp_path: Path) -> None:
    """Jiný seed splitu dá jiný split_seed a aspoň u některého seedu jiný otisk."""
    manifests = [
        _manifest(tmp_path, 0, f"split{seed}", split_seed=seed) for seed in range(6)
    ]

    assert [item.split_seed for item in manifests] == list(range(6))
    assert len({item.split_sha256 for item in manifests}) > 1


def _manifest_bytes(root: Path, seed: int, name: str, split_seed: int = 0) -> bytes:
    return _successful_run(root, seed, name, split_seed).manifest.read_bytes()


def test_manifest_is_byte_identical_for_same_inputs(tmp_path: Path) -> None:
    """Dva běhy se stejnými vstupy dají bajtově totožný manifest."""
    assert _manifest_bytes(tmp_path, 0, "first") == _manifest_bytes(
        tmp_path, 0, "second"
    )


def test_seed_changes_manifest(tmp_path: Path) -> None:
    """Jiný seed modelu dá jiný seed i hash configu, ale týž split."""
    first = RunManifest.model_validate_json(_manifest_bytes(tmp_path, 0, "a"))
    other = RunManifest.model_validate_json(_manifest_bytes(tmp_path, 1, "b"))

    assert (first.seed, other.seed) == (0, 1)
    assert first.config_sha256 != other.config_sha256
    assert first.split_sha256 == other.split_sha256


def test_unreadable_revision_is_refused_before_training(tmp_path: Path) -> None:
    """Revize, kterou nelze přečíst, krok odmítne a nevznikne nic."""
    notes, config, split = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs, revision=missing_revision)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
