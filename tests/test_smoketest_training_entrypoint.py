"""Vstupní bod kroku train_smoketest_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import hashlib
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
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.model_support import (
    OVERLAPPING,
    separable_notes,
    training_config,
    write_raw_split,
    write_split,
)
from tests.support import REVISION, fixed_revision, missing_revision

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mediparse.domain.environment import SourceRevision

PATIENTS: Final = 12


def _inputs(
    root: Path,
    *,
    diagnosis: str = "diabetes",
    seed: int = 0,
    row_id: str | None = "row",
) -> tuple[Path, Path, Path]:
    corpus = separable_notes(PATIENTS, OVERLAPPING)
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(corpus)
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
    split = root / "splits.parquet"
    write_split(split, corpus)
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
    assert written.split_sha256 == ParquetSplitTable(split).read().split.sha256()
    assert written.row_id == "row"
    assert written.environment.source == REVISION
    assert written.model_revision is None
    assert written.tokenizer_revision is None
    assert written.model == ESTIMATOR
    assert tuple(written.environment.packages) == smoketest_training.PACKAGES


@pytest.mark.parametrize(
    ("index", "missing"),
    [(0, "none.parquet"), (1, "none.json"), (2, "none-splits.parquet")],
    ids=["table", "config", "split"],
)
def test_missing_input_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, index: int, missing: str
) -> None:
    """Chybějící tabulka, config tréninku nebo soubor splitu krok odmítne, nic nevznikne a log jmenuje soubor."""
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


def test_split_with_subject_in_two_folds_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pacient ve dvou foldech krok odmítne hláškou o úniku a bez jeho id v logu.

    Otisk sedí podle kontraktu ``Split.sha256``. Bez invariantu by adaptér soubor
    přijal a krok by skončil REFUSED jinou hláškou; rozhoduje aserce na hlášku úniku.
    """
    notes, config, split = _inputs(tmp_path)
    rows = [(90000001, 0), (90000001, 1), (90000002, 1)]
    lines = "".join(f"{subject}\t{fold}\n" for subject, fold in rows)
    write_raw_split(
        split,
        rows,
        {"folds": 2, "seed": 0, "sha256": hashlib.sha256(lines.encode()).hexdigest()},
    )
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "ve více foldech: 1" in caplog.text
    assert "90000001" not in caplog.text
    assert "90000002" not in caplog.text


def test_manifest_split_comes_from_split_file(tmp_path: Path) -> None:
    """Seed, počet foldů a otisk manifestu jsou ze souboru splitu, ne z configu."""
    corpus = separable_notes(PATIENTS, OVERLAPPING)
    notes, config, split = _inputs(tmp_path)
    shared = write_split(split, corpus, seed=5, folds=3)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    written = RunManifest.model_validate_json(
        outputs.manifest.read_text(encoding="utf-8")
    )
    assert code is ExitCode.OK
    assert (written.split_seed, written.split_folds) == (5, 3)
    assert written.split_sha256 == shared.split.sha256()


def _successful_run(root: Path, seed: int, name: str) -> _Outputs:
    notes, config, split = _inputs(root, seed=seed)
    outputs = _Outputs.under(root / name)
    assert _run(notes, config, split, outputs) is ExitCode.OK
    return outputs


def _manifest_bytes(root: Path, seed: int, name: str) -> bytes:
    return _successful_run(root, seed, name).manifest.read_bytes()


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
