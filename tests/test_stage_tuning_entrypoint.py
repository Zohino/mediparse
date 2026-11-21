"""Syntetický průchod nástrojem, nezávislé metriky a provenance."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from sklearn.metrics import f1_score

from mediparse.domain.environment import GpuDevice
from mediparse.domain.run_manifest import StageTuningManifest
from mediparse.domain.stage_tuning import StageLsa
from mediparse.entrypoints import stage_tuning
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure import sklearn_stage
from mediparse.infrastructure.hf_encoder import ENCODER_PACKAGES, HfMeanEncoder
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from mediparse.infrastructure.stage_config import load_stage_config
from tests.hf_encoder_support import SPEC, encoder_parts
from tests.screening_support import SHARED, notes
from tests.stage_support import StepClock, stage_config
from tests.support import REPOSITORY, fixed_revision, mark_synthetic, repository_corpus

if TYPE_CHECKING:
    from collections.abc import Callable
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
        embeddings=root / "embeddings",
        scratch=root / "scratch",
    )


def _run(paths: stage_tuning.StageTuningPaths) -> ExitCode:
    return stage_tuning.run(
        paths=paths,
        environ={"CODEX_THREAD_ID": "test"},
        revision=fixed_revision,
        clock=StepClock(),
    )


def test_end_to_end_metrics_predictions_manifest_and_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 ze skutečných predikcí souhlasí s tabulkou i vítězem v manifestu."""
    original = sklearn_stage.fit_converged
    calls = 0

    def fit_once(fit: Callable[[], object]) -> bool:
        nonlocal calls
        calls += 1
        return original(fit) and calls % 60 != 6

    monkeypatch.setattr(sklearn_stage, "fit_converged", fit_once)
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    trials = pq.read_table(paths.out).to_pylist()
    predictions = pq.read_table(paths.predictions).to_pylist()
    written = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    assert written.completed_trials == written.trials == 6
    assert set(written.environment.packages) == {
        "python",
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "fasttext",
        "optuna",
        "pyarrow",
        "pydantic",
    }
    assert written.seconds == pytest.approx(6.0)
    features = defaultdict(set)
    for row in trials:
        assert type(row["features"]) is int
        assert row["features"] > 0
        features[row["trial"], row["fold"]].add(row["features"])
    assert all(len(counts) == 1 for counts in features.values())
    affected = {row["trial"] for row in trials if not row["converged"]}
    assert affected == {0}
    assert written.nonconverged_trials == len(affected)
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
        "embeddings",
        "scratch",
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


def test_manifest_rejects_inconsistent_run(tmp_path: Path) -> None:
    """Manifest odmítne nevyčerpaný rozpočet a nesoulad předzpracování."""
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    data = json.loads(paths.manifest.read_bytes())
    for changes in ({"completed_trials": 5}, {"text_preprocessing": "dictionary"}):
        with pytest.raises(ValidationError):
            StageTuningManifest.model_validate_json(json.dumps(data | changes))


def test_encoder_stage_records_encoder_provenance(tmp_path: Path) -> None:
    """Syntetický encoder zapíše specifikaci i verze knihoven do manifestu."""
    pytest.importorskip("torch")
    paths = _paths(tmp_path)
    base = stage_config().model_dump(mode="json")
    config = stage_config(
        representations=[SPEC.model_dump(mode="json")],
        families=base["families"][:1],
        trials=2,
    )
    paths.config.write_text(config.model_dump_json(), encoding="utf-8")
    factory = Mock(
        return_value=HfMeanEncoder(
            *encoder_parts(
                sorted({word for n in notes(visits=10) for word in n.text.split()})
            ),
            SPEC.window,
            SPEC.device,
        )
    )
    assert (
        stage_tuning.run(
            paths=paths,
            environ={"CODEX_THREAD_ID": "test"},
            revision=fixed_revision,
            clock=StepClock(),
            encoders=factory,
        )
        is ExitCode.OK
    )
    manifest = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    assert manifest.encoders == (SPEC,)
    assert set(ENCODER_PACKAGES) <= manifest.environment.packages.keys()
    assert "fasttext" in manifest.environment.packages


@pytest.mark.parametrize("field", ["scratch", "embeddings"])
def test_work_paths_inside_checkpoint_refused(tmp_path: Path, field: str) -> None:
    """Pracovní adresáře nesmí vytvořit neúplný checkpoint."""
    paths = _paths(tmp_path)
    paths = replace(paths, **{field: paths.checkpoint / field})
    assert _run(paths) is ExitCode.REFUSED
    assert not any(
        p.exists()
        for p in (paths.out, paths.predictions, paths.manifest, paths.checkpoint)
    )


def test_manifest_without_encoders_validates_as_empty(tmp_path: Path) -> None:
    """Manifest ve starším tvaru bez encoderů se načte s prázdnou n-ticí."""
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    manifest = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    data = manifest.model_dump(mode="json")
    del data["encoders"]
    assert StageTuningManifest.model_validate_json(json.dumps(data)).encoders == ()


@pytest.mark.parametrize("stage", ["classical", "embeddings", "networks"])
def test_repository_stage_configs(stage: str) -> None:
    """Produkční stupně deklarují schválené větve, rozpočty a výchozí body."""
    config = load_stage_config(REPOSITORY / f"config/stage_{stage}.json")
    assert config.seed == 0
    budget = (
        config.trials,
        config.random_startup_trials,
        config.subsample_parts,
        config.folds,
    )
    if stage == "classical":
        assert len(config.branches) == 10
        assert budget == (60, 10, 3, 3)
        assert config.lsa == StageLsa(components=100, max_features=100000)
        return
    assert budget == (20, 5, 3, 3)
    assert config.lsa is None
    if stage == "embeddings":
        assert [b.name for b in config.branches] == ["mmbert-base-mean/logreg"]
        assert config.encoders[0].revision == "c5955035435e2bf121cde7f3c8863ef52ff35d82"
        assert config.families[0].default == {"C": 1.0, "class_weight": None}
    else:
        assert [b.name for b in config.branches] == ["word-tokens/fasttext"]
        assert config.families[0].default == {
            "lr": 0.1,
            "epoch": 5,
            "word_ngrams": 1,
            "dim": 100,
        }


def test_manifest_records_monitored_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest obsahuje zařízení z monitoru s viditelností předanou prostředím."""
    device = GpuDevice(name="synthetic", memory_total_bytes=1024)
    monitor = Mock(return_value=nullcontext(SimpleNamespace(devices=(device,))))
    monkeypatch.setattr(stage_tuning, "gpu_monitor", monitor)
    paths = _paths(tmp_path)
    assert (
        stage_tuning.run(
            paths=paths,
            environ={
                "CODEX_THREAD_ID": "test",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
            revision=fixed_revision,
            clock=StepClock(),
        )
        is ExitCode.OK
    )
    assert StageTuningManifest.model_validate_json(
        paths.manifest.read_bytes()
    ).hardware.gpus == (device,)
    monitor.assert_called_once_with("0", "PCI_BUS_ID")
