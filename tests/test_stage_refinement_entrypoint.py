"""Syntetický průchod doladěním a odmítnutí cizího předkola."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from mediparse.domain.run_manifest import StageRefinementManifest
from mediparse.domain.stage_tuning import StageTuningConfig
from mediparse.entrypoints import stage_refinement, stage_tuning
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from mediparse.infrastructure.stage_trials_table import read_trial_records
from tests.screening_support import SHARED, notes
from tests.stage_support import StepClock, stage_config
from tests.support import (
    REPOSITORY_MATRIX,
    fixed_revision,
    mark_synthetic,
    repository_corpus,
)

if TYPE_CHECKING:
    from pathlib import Path


def _paths(root: Path) -> stage_refinement.StageRefinementPaths:
    mark_synthetic(root)
    corpus = repository_corpus(root / "cohort.parquet", root / "split.parquet")
    ParquetNoteTable(corpus.cohort).write(notes(visits=10))
    ParquetSplitTable(corpus.split).write(SHARED)
    data = stage_config(random_startup_trials=3, trials=8).model_dump(mode="json")
    data["families"] = [
        {
            "family": "linear_svm",
            "space": {
                "params": [
                    {
                        "kind": "float",
                        "name": "C",
                        "low": 0.5,
                        "high": 0.51,
                        "log": True,
                    }
                ]
            },
            "default": {"C": 0.5},
        },
        data["families"][0],
    ]
    config = root / "stage.json"
    config.write_text(
        StageTuningConfig.model_validate_json(json.dumps(data)).model_dump_json(),
        encoding="utf-8",
    )
    screening = stage_tuning.StageTuningPaths(
        corpus=corpus,
        config=config,
        out=root / "screening/trials",
        predictions=root / "screening/predictions",
        checkpoint=root / "screening/parts",
        manifest=root / "screening/manifest",
        embeddings=root / "embeddings",
        scratch=root / "scratch",
    )
    assert (
        stage_tuning.run(
            environ={"CODEX_THREAD_ID": "test"},
            paths=screening,
            revision=fixed_revision,
            clock=StepClock(),
        )
        is ExitCode.OK
    )
    refinement = root / "refinement.json"
    refinement.write_text(
        json.dumps({
            "folds": 2,
            "random_startup_trials": 1,
            "widened": {"linear_svm.C": {"high": 100.0}},
        }),
        encoding="utf-8",
    )
    matrix = root / "matrix.json"
    matrix_data = json.loads(REPOSITORY_MATRIX.read_text(encoding="utf-8"))
    for row in matrix_data["rows"]:
        row["tuning_trials"] = 6
    matrix.write_text(json.dumps(matrix_data), encoding="utf-8")
    return stage_refinement.StageRefinementPaths(
        corpus=corpus,
        config=config,
        stage_results=screening.out,
        stage_manifest=screening.manifest,
        refinement=refinement,
        matrix=matrix,
        out=root / "refined/trials",
        predictions=root / "refined/predictions",
        checkpoint=root / "refined/parts",
        manifest=root / "refined/manifest",
        embeddings=root / "embeddings",
        scratch=root / "scratch",
    )


def _run(paths: stage_refinement.StageRefinementPaths) -> ExitCode:
    return stage_refinement.run(
        environ={"CODEX_THREAD_ID": "test"},
        paths=paths,
        revision=fixed_revision,
        clock=StepClock(),
    )


def test_refinement_uses_widened_space_budget_sampler_and_lineage(
    tmp_path: Path,
) -> None:
    """Rozšíření se projeví v návrzích, rozpočet i sampler pocházejí z doladění."""
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    manifest = StageRefinementManifest.model_validate_json(paths.manifest.read_bytes())
    records = read_trial_records(paths.out)
    assert manifest.completed_trials == manifest.tuning_trials + 1 == 7
    assert manifest.winner_branch == "r1/linear_svm"
    assert any(
        isinstance(r.params["linear_svm.C"], float) and r.params["linear_svm.C"] > 0.51
        for r in records[2:]
    )
    assert (
        manifest.sampler
        == "optuna.samplers.TPESampler(n_startup_trials=2, multivariate=True, group=True)"
    )
    assert manifest.random_startup_trials == 1
    for field, path in (
        ("dataset_sha256", paths.corpus.cohort),
        ("stage_config_sha256", paths.config),
        ("refinement_config_sha256", paths.refinement),
        ("matrix_sha256", paths.matrix),
        ("stage_results_sha256", paths.stage_results),
        ("stage_manifest_sha256", paths.stage_manifest),
        ("results_sha256", paths.out),
        ("predictions_sha256", paths.predictions),
    ):
        assert getattr(manifest, field) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.split_sha256 == SHARED.split.sha256()
    assert manifest.development_notes == 200
    assert [r.default for r in records] == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert manifest.best_params == records[manifest.best_trial].params
    assert manifest.best_macro_f1 == max(r.value for r in records)
    with pytest.raises(ValidationError, match="vítězn"):
        type(manifest).model_validate(
            manifest.model_dump() | {"best_params": dict(manifest.candidates[1].params)}
        )
    before = paths.out.read_bytes(), paths.predictions.read_bytes()
    assert _run(paths) is ExitCode.OK
    assert before == (paths.out.read_bytes(), paths.predictions.read_bytes())


@pytest.mark.parametrize(
    "field",
    [
        "results_sha256",
        "config_sha256",
        "dataset_sha256",
        "split_sha256",
        "text_normalization",
        "best_trial",
    ],
)
def test_refinement_refuses_foreign_screening_before_outputs(
    tmp_path: Path,
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Každá neshoda předkola zabrání fitu i vzniku výstupů."""
    paths = _paths(tmp_path)
    if field == "text_normalization":
        candidates = tmp_path / "candidates.json"
        data = json.loads(paths.corpus.candidates.read_text(encoding="utf-8"))
        data["candidates"]["diabetes"]["acronyms_en"].append("SYNTHETIC")
        candidates.write_text(json.dumps(data), encoding="utf-8")
        paths = replace(paths, corpus=replace(paths.corpus, candidates=candidates))
    else:
        manifest = json.loads(paths.stage_manifest.read_text(encoding="utf-8"))
        manifest[field] = 999 if field == "best_trial" else "f" * 64
        paths.stage_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    scorer = Mock()
    monkeypatch.setattr(stage_refinement, "stage_scorer", scorer)
    assert _run(paths) is ExitCode.REFUSED
    assert field in caplog.text
    scorer.assert_not_called()
    assert not any(
        p.exists()
        for p in (
            paths.out,
            paths.predictions,
            paths.manifest,
            paths.checkpoint,
            paths.out.parent,
        )
    )


def test_main_passes_all_required_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI předá cesty, hodiny a výchozí počet procesů."""
    names = (
        "cohort",
        "split",
        "candidates",
        "lexicon",
        "config",
        "stage-results",
        "stage-manifest",
        "refinement",
        "matrix",
        "out",
        "predictions",
        "checkpoint",
        "manifest",
        "embeddings",
        "scratch",
    )
    argv = ["mediparse-stage-refinement"]
    for name in names:
        argv.extend((f"--{name}", str(tmp_path / name)))
    monkeypatch.setattr("sys.argv", argv)
    run = Mock(return_value=ExitCode.REFUSED)
    monkeypatch.setattr(stage_refinement, "run", run)
    assert stage_refinement.main() is ExitCode.REFUSED
    assert run.call_args.kwargs["jobs"] == 3
    paths = run.call_args.kwargs["paths"]
    for name in names:
        owner = (
            paths.corpus
            if name in {"cohort", "split", "candidates", "lexicon"}
            else paths
        )
        assert getattr(owner, name.replace("-", "_")) == tmp_path / name
