"""Celý krok ladění nad označenou syntetikou: sqlite, determinismus a provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from random import Random
from typing import TYPE_CHECKING
from unittest.mock import Mock

import optuna
import pytest
from optuna.storages import RDBStorage
from sqlalchemy.pool import NullPool

from mediparse.domain.run_manifest import TuningManifest
from mediparse.entrypoints import model_tuning
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.model_support import write_split
from tests.screening_support import notes
from tests.support import (
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
    REVISION,
    fixed_revision,
    mark_synthetic,
)
from tests.tuning_support import matrix_row, tuning_config

if TYPE_CHECKING:
    from pathlib import Path

    from optuna.trial import TrialState

    from mediparse.domain.tuning import ParamValue


@dataclass(frozen=True)
class _Inputs:
    cohort: Path
    split: Path
    config: Path
    matrix: Path


def _inputs(
    root: Path,
    *,
    visits: int = 10,
    marked: bool = True,
    config_changes: dict[str, object] | None = None,
    row_changes: dict[str, object] | None = None,
) -> _Inputs:
    corpus = notes(visits=visits)
    paths = _Inputs(
        root / "cohort.parquet",
        root / "split.parquet",
        root / "tuning.json",
        root / "matrix.json",
    )
    ParquetNoteTable(paths.cohort).write(corpus)
    write_split(paths.split, corpus, seed=23)
    paths.config.write_text(
        tuning_config(**(config_changes or {})).model_dump_json(), encoding="utf-8"
    )
    paths.matrix.write_text(
        json.dumps({
            "rows": [matrix_row(**(row_changes or {})).model_dump(mode="json")]
        }),
        encoding="utf-8",
    )
    if marked:
        mark_synthetic(root)
    return paths


def _run(inputs: _Inputs, out: Path) -> ExitCode:
    return model_tuning.run(
        environ={"CODEX_THREAD_ID": "test"},
        cohort=inputs.cohort,
        split=inputs.split,
        config=inputs.config,
        matrix=inputs.matrix,
        candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
        lexicon=REPOSITORY_SECTION_HEADERS,
        storage=out / "tuning.sqlite3",
        manifest=out / "tuning_manifest.json",
        revision=fixed_revision,
        clock=Mock(now=Mock(side_effect=[10.0, 12.5])),
    )


def _manifest(out: Path) -> TuningManifest:
    return TuningManifest.model_validate_json(
        (out / "tuning_manifest.json").read_text(encoding="utf-8")
    )


def test_tuning_manifest_records_budget_inputs_training_and_costs(
    tmp_path: Path,
) -> None:
    """Rozpočet 3 je ručně zapsaný do matice, čas 2,5 s dává injektovaný čítač."""
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.OK
    manifest = _manifest(out)
    expected_budget = 3
    assert manifest.completed_trials + manifest.pruned_trials == expected_budget
    assert manifest.tuning_trials == expected_budget
    assert manifest.space == tuning_config().space
    assert (manifest.row_id, manifest.fold, manifest.seed) == ("en-svm", 0, 7)
    assert (
        manifest.dataset_sha256
        == hashlib.sha256(inputs.cohort.read_bytes()).hexdigest()
    )
    assert (
        manifest.config_sha256 == hashlib.sha256(inputs.config.read_bytes()).hexdigest()
    )
    assert (
        manifest.matrix_sha256 == hashlib.sha256(inputs.matrix.read_bytes()).hexdigest()
    )
    shared = ParquetSplitTable(inputs.split).read()
    holdout = shared.split.holdout(ParquetNoteTable(inputs.cohort).read(), 0)
    assert manifest.split_sha256 == shared.split.sha256()
    assert (manifest.split_seed, manifest.split_folds) == (23, 2)
    assert manifest.train_notes == len(holdout.train)
    assert manifest.train_patients == len({note.subject_id for note in holdout.train})
    assert manifest.seconds == pytest.approx(2.5)
    assert manifest.hardware.gpus == ()
    assert manifest.environment.source == REVISION
    assert "optuna" in manifest.environment.packages
    assert (
        manifest.sampler
        == "optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True)"
    )
    assert manifest.pruner == "optuna.pruners.MedianPruner"
    assert (out / "tuning.sqlite3").is_file()


@pytest.mark.parametrize("rewrite_excluded", [False, True])
def test_repeated_search_is_independent_of_outer_test_and_development(
    tmp_path: Path, *, rewrite_excluded: bool
) -> None:
    """Dva nové sqlite opakují výsledek i po přepsání všech vyloučených textů."""
    inputs = _inputs(tmp_path)
    table = ParquetNoteTable(inputs.cohort)
    corpus = table.read()
    donors = Random(17).sample(corpus, len(corpus))
    table.write([
        replace(note, text=donor.text) if index % 3 == 0 else note
        for index, (note, donor) in enumerate(zip(corpus, donors, strict=True))
    ])
    first, second = tmp_path / "first", tmp_path / "second"
    assert _run(inputs, first) is ExitCode.OK
    if rewrite_excluded:
        shared = ParquetSplitTable(inputs.split).read()
        excluded = shared.split.folds[0] | shared.split.development
        table = ParquetNoteTable(inputs.cohort)
        table.write([
            replace(note, text="synthetic excluded entirely different vocabulary")
            if note.subject_id in excluded
            else note
            for note in table.read()
        ])
    assert _run(inputs, second) is ExitCode.OK
    first_trials, second_trials = _trial_history(first), _trial_history(second)
    minimum_distinct_values = 2
    assert (
        len({value for _, value, _ in first_trials if value is not None})
        >= minimum_distinct_values
    )
    assert second_trials == first_trials
    before, after = _manifest(first), _manifest(second)
    assert after.best == before.best
    assert after.best_macro_f1 == before.best_macro_f1
    assert (after.dataset_sha256 != before.dataset_sha256) is rewrite_excluded


def _trial_history(
    out: Path,
) -> list[tuple[dict[str, ParamValue], float | None, TrialState]]:
    storage = RDBStorage(
        f"sqlite:///{out / 'tuning.sqlite3'}", engine_kwargs={"poolclass": NullPool}
    )
    study = optuna.load_study(study_name="model-tuning", storage=storage)
    return [(trial.params, trial.value, trial.state) for trial in study.trials]


def test_existing_sqlite_is_refused_without_overwriting_manifest(
    tmp_path: Path,
) -> None:
    """Rerun nepřičítá pokusy ke stejné studii ani nepřepíše úspěšný manifest."""
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.OK
    original = (out / "tuning_manifest.json").read_bytes()
    assert _run(inputs, out) is ExitCode.REFUSED
    assert (out / "tuning_manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    ("config_changes", "row_changes", "visits", "marked", "message"),
    [
        ({"row_id": "absent"}, {}, 10, True, "není v experimentální matici"),
        ({}, {"model": "logreg"}, 10, True, "zatím není scorer"),
        (
            {
                "space": {
                    "params": [{"kind": "categorical", "name": "C", "choices": [1.0]}]
                }
            },
            {},
            10,
            True,
            "parametrům scoreru",
        ),
        ({}, {}, 1, True, "zpráv, metriky potřebují v každé třídě aspoň 10"),
        ({}, {}, 10, False, "MIMIC"),
    ],
)
# Signatura parametrizace rozlišuje config, řádek, velikost kohorty a stráž.
def test_invalid_tuning_is_refused(  # ruff: ignore[too-many-arguments]
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    config_changes: dict[str, object],
    row_changes: dict[str, object],
    visits: int,
    *,
    marked: bool,
    message: str,
) -> None:
    """Chybějící řádek, rodina, překlep, MIN_CELL i stráž agenta mají doménové odmítnutí."""
    inputs = _inputs(
        tmp_path,
        visits=visits,
        marked=marked,
        config_changes=config_changes,
        row_changes=row_changes,
    )
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.REFUSED
    assert message in caplog.text
    assert not (out / "tuning_manifest.json").exists()


def test_main_passes_workflow_paths_and_runtime_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Composition root přenese cesty pravidla do testovatelného run a nastaví log."""
    paths = {
        name: tmp_path / name
        for name in (
            "cohort",
            "config",
            "matrix",
            "candidates",
            "lexicon",
            "split",
            "storage",
            "manifest",
            "log",
        )
    }
    run = Mock(return_value=ExitCode.OK)
    logging = Mock()
    monkeypatch.setattr(model_tuning, "run", run)
    monkeypatch.setattr(model_tuning, "configure_logging", logging)
    assert model_tuning.main(**paths) is ExitCode.OK
    logging.assert_called_once_with(paths["log"])
    assert {key: run.call_args.kwargs[key] for key in paths if key != "log"} == {
        key: value for key, value in paths.items() if key != "log"
    }
    assert callable(run.call_args.kwargs["revision"])
    assert callable(run.call_args.kwargs["clock"].now)
