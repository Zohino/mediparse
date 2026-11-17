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

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.run_manifest import TuningManifest
from mediparse.entrypoints import model_tuning
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_corpus import ModelCorpusPaths
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.model_support import ACRONYM_VOCABULARY, model_input, write_split
from tests.screening_support import notes
from tests.support import (
    REPOSITORY_SECTION_HEADERS,
    REVISION,
    fixed_revision,
    mark_synthetic,
    repository_corpus,
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
    mark_synthetic(root)
    return paths


def _run(inputs: _Inputs, out: Path) -> ExitCode:
    return model_tuning.run(
        paths=model_tuning.TuningPaths(
            corpus=repository_corpus(inputs.cohort, inputs.split),
            config=inputs.config,
            matrix=inputs.matrix,
            storage=out / "tuning.sqlite3",
            manifest=out / "tuning_manifest.json",
        ),
        environ={"CODEX_THREAD_ID": "test"},
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
    ("config_changes", "row_changes", "visits", "message"),
    [
        ({"row_id": "absent"}, {}, 10, "není v experimentální matici"),
        ({}, {"model": "logreg"}, 10, "zatím není scorer"),
        (
            {
                "space": {
                    "params": [{"kind": "categorical", "name": "C", "choices": [1.0]}]
                }
            },
            {},
            10,
            "parametrům scoreru",
        ),
        ({}, {}, 1, "zpráv, metriky potřebují v každé třídě aspoň 10"),
    ],
)
# Signatura parametrizace rozlišuje config, řádek a velikost kohorty.
def test_invalid_tuning_is_refused(  # ruff: ignore[too-many-arguments]
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    config_changes: dict[str, object],
    row_changes: dict[str, object],
    visits: int,
    *,
    message: str,
) -> None:
    """Chybějící řádek, rodina, překlep i MIN_CELL mají doménové odmítnutí."""
    inputs = _inputs(
        tmp_path,
        visits=visits,
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
    paths = model_tuning.TuningPaths(
        corpus=ModelCorpusPaths(
            cohort=tmp_path / "cohort",
            candidates=tmp_path / "candidates",
            lexicon=tmp_path / "lexicon",
            split=tmp_path / "split",
        ),
        config=tmp_path / "config",
        matrix=tmp_path / "matrix",
        storage=tmp_path / "storage",
        manifest=tmp_path / "manifest",
    )
    log = tmp_path / "log"
    run = Mock(return_value=ExitCode.OK)
    logging = Mock()
    monkeypatch.setattr(model_tuning, "run", run)
    monkeypatch.setattr(model_tuning, "configure_logging", logging)
    assert model_tuning.main(paths=paths, log=log) is ExitCode.OK
    logging.assert_called_once_with(log)
    assert run.call_args.kwargs["paths"] == paths
    assert callable(run.call_args.kwargs["revision"])
    assert callable(run.call_args.kwargs["clock"].now)


@pytest.mark.parametrize("preprocessing", ["normalized", "dictionary"])
def test_manifest_preserves_preprocessing_and_strip(
    tmp_path: Path, preprocessing: str
) -> None:
    """Manifest ladění nese strip, slovník a explicitní úroveň předzpracování."""
    inputs = _inputs(
        tmp_path,
        row_changes={"input": model_input(strip=True, preprocessing=preprocessing)},
    )
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.OK
    manifest = _manifest(out)
    assert manifest.diagnostic_strip is not None
    assert manifest.diagnostic_strip.sections == ("discharge_diagnosis",)
    assert (
        manifest.diagnostic_strip.lexicon_sha256
        == hashlib.sha256(REPOSITORY_SECTION_HEADERS.read_bytes()).hexdigest()
    )
    assert manifest.text_preprocessing.value == preprocessing
    assert manifest.text_normalization is not None
    assert (
        manifest.text_normalization.acronyms["COPD"]
        == "chronic obstructive pulmonary disease"
    )
    assert bool(manifest.text_normalization.variants) == (preprocessing == "dictionary")


def test_normalization_changes_tuning_scores(tmp_path: Path) -> None:
    """Sjednocení zkratky a názvu odstraní signál syntetických tříd a změní skóre."""
    scores = []
    for preprocessing in ("none", "normalized"):
        root = tmp_path / preprocessing
        inputs = _inputs(
            root, row_changes={"input": model_input(preprocessing=preprocessing)}
        )
        table = ParquetNoteTable(inputs.cohort)
        table.write([
            replace(
                note,
                text=ACRONYM_VOCABULARY[0 if note.has_label(Diagnosis.DIABETES) else 1],
            )
            for note in table.read()
        ])
        out = root / "out"
        assert _run(inputs, out) is ExitCode.OK
        scores.append(_manifest(out).best_macro_f1)

    assert scores[0] != scores[1]
