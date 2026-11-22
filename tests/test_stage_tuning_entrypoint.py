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
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.representation import ClassifierFamily, Word2VecRepresentation
from mediparse.domain.run_manifest import StageTuningManifest
from mediparse.domain.stage_tuning import StageLsa
from mediparse.entrypoints import stage_tuning
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure import sklearn_stage
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
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


def _run(paths: stage_tuning.StageTuningPaths, jobs: int = 1) -> ExitCode:
    return stage_tuning.run(
        paths=paths,
        environ={"CODEX_THREAD_ID": "test"},
        revision=fixed_revision,
        clock=StepClock(),
        jobs=jobs,
    )


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.usefixtures("loky_executor")
def test_end_to_end_metrics_predictions_manifest_and_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int
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
    assert _run(paths, jobs=jobs) is ExitCode.OK
    trials = pq.read_table(paths.out).to_pylist()
    predictions = pq.read_table(paths.predictions).to_pylist()
    written = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    assert written.jobs == jobs
    assert written.completed_trials == written.trials == 6
    assert set(written.environment.packages) == {
        "python",
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "fasttext",
        "gensim",
        "joblib",
        "threadpoolctl",
        "optuna",
        "pyarrow",
        "pydantic",
    }
    assert written.seconds == pytest.approx(6.0)
    n_features = defaultdict(set)
    for row in trials:
        assert type(row["n_features"]) is int
        assert row["n_features"] > 0
        n_features[row["trial"], row["fold"]].add(row["n_features"])
    assert all(len(counts) == 1 for counts in n_features.values())

    affected = {row["trial"] for row in trials if not row["converged"]}
    # Monkeypatch fit_converged do procesů loky nedosáhne; jejich nekonvergenci
    # ověřuje test_parallel_fits_report_nonconvergence skutečným limitem iterací.
    if jobs == 1:
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
    assert _run(paths, jobs=jobs) is ExitCode.OK
    assert before == (paths.out.read_bytes(), paths.predictions.read_bytes())
    assert not paths.checkpoint.exists()


def test_foreign_checkpoint_refused_before_outputs(tmp_path: Path) -> None:
    """Cizí klíč nesmí vytvořit žádný finální výstup."""
    paths = _paths(tmp_path)
    paths.checkpoint.mkdir(parents=True)
    (paths.checkpoint / "key.json").write_text('{"key":"foreign"}', encoding="utf-8")
    assert _run(paths) is ExitCode.REFUSED
    assert not any(p.exists() for p in (paths.out, paths.predictions, paths.manifest))


@pytest.mark.parametrize(("jobs_args", "jobs"), [([], 3), (["--jobs", "2"], 2)])
def test_main_accepts_all_required_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs_args: list[str], jobs: int
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
    argv.extend(jobs_args)
    monkeypatch.setattr("sys.argv", argv)
    run = Mock(return_value=ExitCode.REFUSED)
    monkeypatch.setattr(stage_tuning, "run", run)
    assert stage_tuning.main() is ExitCode.REFUSED
    assert run.call_args.kwargs["jobs"] == jobs
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


@pytest.mark.parametrize("jobs", [0, 6])
def test_main_rejects_jobs_outside_diagnoses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], jobs: int
) -> None:
    """CLI odmítne počet procesů mimo rozsah diagnóz."""
    monkeypatch.setattr("sys.argv", ["mediparse-stage-tuning", "--jobs", str(jobs)])
    with pytest.raises(SystemExit):
        stage_tuning.main()
    assert "argument --jobs: invalid choice" in capsys.readouterr().err


def test_manifest_rejects_inconsistent_run(tmp_path: Path) -> None:
    """Manifest odmítne nevyčerpaný rozpočet a nesoulad předzpracování."""
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    data = json.loads(paths.manifest.read_bytes())
    for changes, message in (
        ({"jobs": 0}, "greater than 0"),
        ({"completed_trials": 5}, "vyčerpat rozpočet"),
        (
            {"text_preprocessing": {"r1": "none", "r2": "none"}},
            "Úroveň předzpracování none",
        ),
        ({"text_preprocessing": {"r1": "normalized", "r2": "none"}}, "variant"),
        (
            {"text_normalization": {"r1": data["text_normalization"]["r1"]}},
            "stejné reprezentace",
        ),
        ({"text_preprocessing": {}, "text_normalization": {}}, "at least 1 item"),
    ):
        with pytest.raises(ValidationError, match=message):
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
    assert "gensim" in manifest.environment.packages


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
    assert config.strip_diagnostic_sections is False
    assert (
        config.text_preprocessing
        == {
            "classical": {
                "word": TextPreprocessing.DICTIONARY,
                "char_wb": TextPreprocessing.DICTIONARY,
            },
            "embeddings": {
                "mmbert-base-mean": TextPreprocessing.NONE,
                "mmbert-base-mean-dictionary": TextPreprocessing.DICTIONARY,
                "word2vec": TextPreprocessing.DICTIONARY,
            },
            "networks": {"word-tokens": TextPreprocessing.DICTIONARY},
        }[stage]
    )
    assert config.seed == 0
    budget = (
        config.trials,
        config.random_startup_trials,
        config.subsample_parts,
        config.folds,
    )
    if stage == "classical":
        assert len(config.branches) == 10
        assert budget == (60, 10, 3, 5)
        assert config.lsa == StageLsa(components=100, max_features=100000)
        return
    assert config.lsa is None
    if stage == "embeddings":
        assert budget == (54, 5, 3, 5)
        assert [b.name for b in config.branches] == [
            "mmbert-base-mean/logreg",
            "mmbert-base-mean/linear_svm",
            "mmbert-base-mean/lightgbm",
            "mmbert-base-mean-dictionary/logreg",
            "mmbert-base-mean-dictionary/linear_svm",
            "mmbert-base-mean-dictionary/lightgbm",
            "word2vec/logreg",
            "word2vec/linear_svm",
            "word2vec/lightgbm",
        ]
        word2vec = config.representations[2]
        assert isinstance(word2vec, Word2VecRepresentation)
        assert word2vec.default == {
            "vector_size": 100,
            "window": 5,
            "min_count": 5,
            "sg": 0,
            "epochs": 5,
            "negative": 5,
            "sample": 0.001,
        }
        assert word2vec.workers == 16
        assert word2vec.space.model_dump(mode="json") == {
            "params": [
                {
                    "kind": "categorical",
                    "name": "vector_size",
                    "choices": [100, 200, 300],
                },
                {"kind": "int", "name": "window", "low": 2, "high": 10, "log": False},
                {"kind": "int", "name": "min_count", "low": 1, "high": 20, "log": True},
                {"kind": "categorical", "name": "sg", "choices": [0, 1]},
                {"kind": "int", "name": "epochs", "low": 3, "high": 10, "log": False},
                {"kind": "int", "name": "negative", "low": 1, "high": 20, "log": True},
                {
                    "kind": "float",
                    "name": "sample",
                    "low": 1e-5,
                    "high": 1e-2,
                    "log": True,
                },
            ]
        }
        classical = load_stage_config(REPOSITORY / "config/stage_classical.json")
        by_family = {family.family: family for family in config.families}
        classical_by_family = {family.family: family for family in classical.families}
        for family, source in (
            (ClassifierFamily.LINEAR_SVM, ClassifierFamily.LINEAR_SVM),
            (ClassifierFamily.LIGHTGBM, ClassifierFamily.LIGHTGBM_LSA),
        ):
            assert by_family[family].space == classical_by_family[source].space
            assert by_family[family].default == classical_by_family[source].default
        assert config.encoders[0].revision == "c5955035435e2bf121cde7f3c8863ef52ff35d82"
        assert config.families[0].default == {"C": 1.0, "class_weight": None}
    else:
        assert budget == (20, 5, 3, 5)
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


def test_manifest_preserves_diagnostic_strip(tmp_path: Path) -> None:
    """Zapnutý strip stupně zapíše diagnostické sekce i otisk lexikonu."""
    paths = _paths(tmp_path)
    paths.config.write_text(
        stage_config(strip_diagnostic_sections=True).model_dump_json(), encoding="utf-8"
    )
    assert _run(paths) is ExitCode.OK
    manifest = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    assert manifest.diagnostic_strip is not None
    assert manifest.diagnostic_strip.sections == ("discharge_diagnosis",)
    assert (
        manifest.diagnostic_strip.lexicon_sha256
        == hashlib.sha256(paths.corpus.lexicon.read_bytes()).hexdigest()
    )


def test_stage_records_and_passes_normalization_per_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scorer i manifest dostávají stejné slovníky jednotlivých reprezentací."""
    tfidf = Mock(wraps=sklearn_stage.tfidf_vectorizer)
    monkeypatch.setattr(sklearn_stage, "tfidf_vectorizer", tfidf)
    paths = _paths(tmp_path)
    assert _run(paths) is ExitCode.OK
    dictionary = load_text_normalization(paths.corpus.candidates, variants=True)
    assert any(c.args[0] == dictionary for c in tfidf.call_args_list)
    assert any(c.args[0] is None for c in tfidf.call_args_list)
    assert all(
        c.args[0] == dictionary or c.args[0] is None for c in tfidf.call_args_list
    )
    written = StageTuningManifest.model_validate_json(paths.manifest.read_bytes())
    assert written.text_preprocessing == {
        "r1": TextPreprocessing.DICTIONARY,
        "r2": TextPreprocessing.NONE,
    }
    assert written.text_normalization == {"r1": dictionary, "r2": None}
