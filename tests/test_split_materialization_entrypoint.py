"""Vstupní bod kroku make_smoketest_splits: soubor splitu, odmítnutí a determinismus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.entrypoints import split_materialization
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.model_support import (
    OVERLAPPING,
    separable_notes,
    split_config,
    split_config_json,
)

if TYPE_CHECKING:
    from pathlib import Path

PATIENTS: Final = 12


def _inputs(
    root: Path, *, seed: int = 0, patients: int = PATIENTS
) -> tuple[Path, Path]:
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(separable_notes(patients, OVERLAPPING))
    config = root / f"split-{seed}.json"
    config.write_text(json.dumps(split_config_json(seed=seed)), encoding="utf-8")
    return notes, config


def _run(notes: Path, config: Path, output: Path) -> ExitCode:
    return split_materialization.run(notes=notes, config=config, output=output)


def test_writes_split_readable_by_adapter(tmp_path: Path) -> None:
    """Běh zapíše splits.parquet s configem a pacienty všech zpráv."""
    notes, config = _inputs(tmp_path)
    output = tmp_path / "build" / "splits.parquet"

    code = _run(notes, config, output)

    shared = ParquetSplitTable(output).read()
    assert code is ExitCode.OK
    assert shared.config == split_config(seed=0)
    assert frozenset().union(*shared.split.folds) == set(range(1, PATIENTS + 1))


@pytest.mark.parametrize("missing", ["notes", "config"])
def test_missing_input_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, missing: str
) -> None:
    """Chybějící tabulka nebo config krok odmítne, nic nevznikne a log jmenuje soubor."""
    notes, config = _inputs(tmp_path)
    paths = {"notes": notes, "config": config}
    paths[missing] = tmp_path / f"none-{missing}"
    output = tmp_path / "out" / "splits.parquet"

    code = _run(paths["notes"], paths["config"], output)

    assert code is ExitCode.REFUSED
    assert not output.exists()
    assert f"none-{missing}" in caplog.text


def test_invalid_config_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Config s jedním foldem krok odmítne, nic nevznikne a log jmenuje soubor."""
    notes, config = _inputs(tmp_path)
    config.write_text(json.dumps(split_config_json(folds=1)), encoding="utf-8")
    output = tmp_path / "out" / "splits.parquet"

    code = _run(notes, config, output)

    assert code is ExitCode.REFUSED
    assert not output.exists()
    assert config.name in caplog.text


def test_fewer_patients_than_folds_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Méně pacientů než foldů skončí REFUSED s počty, ne tracebackem."""
    notes, config = _inputs(tmp_path, patients=1)
    output = tmp_path / "out" / "splits.parquet"

    code = _run(notes, config, output)

    assert code is ExitCode.REFUSED
    assert not output.exists()
    assert "Pacientů je 1" in caplog.text


def test_same_inputs_give_identical_file(tmp_path: Path) -> None:
    """Dva běhy se stejnými vstupy dají bajtově shodný soubor."""
    notes, config = _inputs(tmp_path)

    assert _run(notes, config, tmp_path / "a.parquet") is ExitCode.OK
    assert _run(notes, config, tmp_path / "b.parquet") is ExitCode.OK

    assert (tmp_path / "a.parquet").read_bytes() == (
        tmp_path / "b.parquet"
    ).read_bytes()


def test_split_seed_changes_fingerprint(tmp_path: Path) -> None:
    """Seed splitu mění přiřazení: aspoň u některého ze 6 seedů je otisk jiný."""
    fingerprints = set()
    for seed in range(6):
        notes, config = _inputs(tmp_path, seed=seed)
        output = tmp_path / f"splits-{seed}.parquet"
        assert _run(notes, config, output) is ExitCode.OK
        shared = ParquetSplitTable(output).read()
        assert shared.config.seed == seed
        fingerprints.add(shared.split.sha256())

    assert len(fingerprints) > 1
