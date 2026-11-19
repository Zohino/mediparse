"""Syntetický průchod nástrojem, nezávislé metriky a provenance."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from sklearn.metrics import f1_score

from mediparse.domain.run_manifest import StageTuningManifest
from mediparse.entrypoints import stage_tuning
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from mediparse.infrastructure.stage_config import load_stage_config
from tests.screening_support import SHARED, notes
from tests.stage_support import StepClock, stage_config
from tests.support import REPOSITORY, fixed_revision, mark_synthetic, repository_corpus

if TYPE_CHECKING:
    from pathlib import Path


def _paths(root: Path) -> stage_tuning.StageTuningPaths:
    cohort, split, config = (
        root / name for name in ("cohort.parquet", "split.parquet", "config.json")
    )
    ParquetNoteTable(cohort).write(notes(visits=10))
    ParquetSplitTable(split).write(SHARED)
    config.write_text(stage_config().model_dump_json(), encoding="utf-8")
    mark_synthetic(root)
    return stage_tuning.StageTuningPaths(
        corpus=repository_corpus(cohort, split),
        config=config,
        out=root / "out/trials.parquet",
        predictions=root / "out/predictions.parquet",
        checkpoint=root / "out/checkpoint",
        manifest=root / "out/manifest.json",
    )


def _run(paths: stage_tuning.StageTuningPaths) -> ExitCode:
    return stage_tuning.run(
        paths=paths,
        environ={"CODEX_THREAD_ID": "test"},
        revision=fixed_revision,
        clock=StepClock(),
    )


def test_end_to_end_metrics_predictions_manifest_and_repeat(tmp_path: Path) -> None:
    """F1 ze skutečných predikcí souhlasí s tabulkou i vítězem v manifestu."""
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    trials = pq.read_table(paths.out).to_pylist()
    predictions = pq.read_table(paths.predictions).to_pylist()
    written = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    assert written.completed_trials == written.trials == 6
    assert written.seconds == pytest.approx(6.0)
    for field, path in (
        ("dataset_sha256", paths.corpus.cohort),
        ("config_sha256", paths.config),
        ("results_sha256", paths.out),
        ("predictions_sha256", paths.predictions),
    ):
        assert getattr(written, field) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert written.split_sha256 == SHARED.split.sha256()
    assert written.sample_notes == written.development_notes == 200
    assert written.sample_patients == written.development_patients == 20
    cells = defaultdict(list)
    for row in predictions:
        cells[row["trial"], row["fold"], row["diagnosis"]].append(row)
    targets = defaultdict(list)
    fold_notes = defaultdict(set)
    for row in trials:
        cell = cells[row["trial"], row["fold"], row["diagnosis"]]
        assert all(p["predicted"] == (p["score"] >= row["threshold"]) for p in cell)
        assert f1_score(
            [p["label"] for p in cell], [p["predicted"] for p in cell]
        ) == pytest.approx(row["max_f1"])
        assert len({p["note_id"] for p in cell}) == len(cell)
        fold_notes[row["fold"]].update(p["note_id"] for p in cell)
        targets[row["trial"]].append(row["max_f1"])
    for row in trials:
        assert row["value"] == pytest.approx(
            sum(targets[row["trial"]]) / len(targets[row["trial"]])
        )
    assert written.best_macro_f1 == next(
        r["value"] for r in trials if r["trial"] == written.best_trial
    )
    expected = {
        n.note_id for n in notes(visits=10) if n.subject_id in SHARED.split.development
    }
    assert Counter(n for fold in fold_notes.values() for n in fold) == Counter(
        dict.fromkeys(expected, 1)
    )
    for (_, fold, _), cell in cells.items():
        assert {p["note_id"] for p in cell} == fold_notes[fold]
    assert len(predictions) == 6 * 5 * len(expected)
    before = paths.out.read_bytes(), paths.predictions.read_bytes()
    assert _run(paths) is ExitCode.OK
    assert before == (paths.out.read_bytes(), paths.predictions.read_bytes())
    assert not paths.checkpoint.exists()


def test_foreign_checkpoint_refused_before_outputs(tmp_path: Path) -> None:
    """Cizí klíč nesmí vytvořit žádný finální výstup."""
    paths = _paths(tmp_path)
    paths.checkpoint.mkdir(parents=True)
    (paths.checkpoint / "key.json").write_text('{"key":"foreign"}', encoding="utf-8")
    assert _run(paths) is ExitCode.REFUSED
    assert not any(p.exists() for p in (paths.out, paths.predictions, paths.manifest))


def test_main_accepts_all_required_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI předá explicitní cesty a hodiny vstupnímu bodu."""
    names = (
        "cohort",
        "split",
        "config",
        "candidates",
        "lexicon",
        "out",
        "predictions",
        "checkpoint",
        "manifest",
    )
    argv = ["mediparse-stage-tuning"]
    for name in names:
        argv.extend((f"--{name}", str(tmp_path / name)))
    monkeypatch.setattr("sys.argv", argv)
    run = Mock(return_value=ExitCode.REFUSED)
    monkeypatch.setattr(stage_tuning, "run", run)
    assert stage_tuning.main() is ExitCode.REFUSED
    paths = run.call_args.kwargs["paths"]
    for name in names:
        assert (
            getattr(
                paths.corpus
                if name in {"cohort", "split", "candidates", "lexicon"}
                else paths,
                name,
            )
            == tmp_path / name
        )
    assert isinstance(run.call_args.kwargs["clock"], stage_tuning.PerfCounterClock)


def test_repository_config_and_manifest_validation(tmp_path: Path) -> None:
    """Produkční config má deset větví a manifest odmítne nevyčerpaný rozpočet."""
    config = load_stage_config(REPOSITORY / "config/stage_classical.json")
    assert len(config.branches) == 10
    assert (
        config.trials,
        config.random_startup_trials,
        config.subsample_parts,
        config.folds,
    ) == (60, 10, 3, 3)
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    data = json.loads(paths.manifest.read_bytes())
    for changes in ({"completed_trials": 5}, {"text_preprocessing": "dictionary"}):
        with pytest.raises(ValidationError):
            StageTuningManifest.model_validate_json(json.dumps(data | changes))
