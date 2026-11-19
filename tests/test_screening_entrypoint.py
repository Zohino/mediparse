"""Nástroj screeningu: skutečné adaptéry, provenance, stráž dat a determinismus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from itertools import product
from random import Random
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import ScreeningManifest
from mediparse.entrypoints import screening
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_corpus import ModelCorpusPaths
from mediparse.infrastructure.hf_encoder import ENCODER_PACKAGES, HfMeanEncoder
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.hf_encoder_support import SPEC, VERSIONS, encoder_parts
from tests.screening_support import DEVELOPMENT, SHARED, notes
from tests.support import (
    REPOSITORY_SECTION_HEADERS,
    REVISION,
    fixed_revision,
    mark_synthetic,
    missing_revision,
    repository_corpus,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from mediparse.domain.environment import SourceRevision

# Vnitřní trénink musí umožnit dělení uzlů při min_child_samples=20;
# konstantní skóre TunedThresholdClassifierCV odmítne.
VISITS_PER_PATIENT = 16
CLASSICAL_GRID = {
    "families": ("complement_nb", "linear_svm"),
    "vectorizers": [{"name": "word", "analyzer": "word", "ngram_range": [1, 1]}],
}


@dataclass(frozen=True)
class _Inputs:
    cohort: Path
    split: Path
    config: Path


def _inputs(
    root: Path,
    *,
    preprocessing: str = "none",
    strip: bool = False,
    grid: Mapping[str, object] = CLASSICAL_GRID,
    visits: int = 2,
) -> _Inputs:
    cohort = root / "cohort.parquet"
    ParquetNoteTable(cohort).write(notes(visits=visits))
    split = root / "split.parquet"
    ParquetSplitTable(split).write(SHARED)
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "seed": 7,
                "folds": 2,
                "threshold_folds": 2,
                "lsa_components": 2,
                "input": {
                    "strip_diagnostic_sections": strip,
                    "text_preprocessing": preprocessing,
                },
            }
            | dict(grid)
        ),
        encoding="utf-8",
    )
    return _Inputs(cohort, split, config)


def _run(
    inputs: _Inputs,
    root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    revision: Callable[[], SourceRevision] = fixed_revision,
) -> ExitCode:
    return screening.run(
        paths=screening.ScreeningPaths(
            corpus=repository_corpus(inputs.cohort, inputs.split),
            config=inputs.config,
            out=root / "classical.parquet",
            predictions=root / "classical_predictions.parquet",
            manifest=root / "classical_manifest.json",
            checkpoint=root / "checkpoint",
            embeddings=root / "embeddings",
            scratch=root / "scratch",
        ),
        environ={} if environ is None else environ,
        revision=revision,
    )


def test_results_manifest_records_actual_inputs_and_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest váže agregáty na skutečné soubory, původní split, prostředí a velikost vývoje."""
    packages = Mock(wraps=screening.installed_versions)
    monkeypatch.setattr(screening, "installed_versions", packages)
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    mark_synthetic(tmp_path)
    assert _run(inputs, out, environ={"CODEX_THREAD_ID": "test"}) is ExitCode.OK
    packages.assert_called_once_with(screening.PACKAGES)
    written = ScreeningManifest.model_validate_json(
        (out / "classical_manifest.json").read_text(encoding="utf-8")
    )
    assert written.encoders == ()
    assert "torch" not in written.environment.packages
    result = out / "classical.parquet"
    assert (
        written.dataset_sha256 == hashlib.sha256(inputs.cohort.read_bytes()).hexdigest()
    )
    assert written.results_sha256 == hashlib.sha256(result.read_bytes()).hexdigest()
    assert (
        written.predictions_sha256
        == hashlib.sha256(
            (out / "classical_predictions.parquet").read_bytes()
        ).hexdigest()
    )
    assert (
        written.config_sha256 == hashlib.sha256(inputs.config.read_bytes()).hexdigest()
    )
    shared = ParquetSplitTable(inputs.split).read()
    assert written.split_sha256 == shared.split.sha256()
    assert written.split_seed == shared.config.seed
    assert written.split_folds == shared.config.folds
    development = [note for note in notes() if note.subject_id in DEVELOPMENT]
    assert written.development_notes == len(development)
    assert written.development_patients == len({
        note.subject_id for note in development
    })
    assert written.seed == 7
    assert written.diagnostic_strip is None
    assert written.text_preprocessing is TextPreprocessing.NONE
    assert written.text_normalization is None
    assert written.environment.source == REVISION
    assert set(written.environment.packages) == {
        "python",
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "fasttext",
        "pyarrow",
        "pydantic",
    }
    assert all(written.environment.packages.values())
    grid = json.loads(inputs.config.read_text(encoding="utf-8"))
    assert (
        pq.read_table(result).num_rows
        == grid["folds"] * len(grid["vectorizers"]) * len(grid["families"]) * 5
    )


@pytest.mark.parametrize(
    "grid",
    [
        CLASSICAL_GRID
        | {
            "families": (
                "complement_nb",
                "logreg",
                "linear_svm",
                "nb_svm",
                "lightgbm_lsa",
            )
        },
        {"families": ("fasttext",), "vectorizers": [{"name": "word-tokens"}]},
    ],
)
def test_repeated_run_has_identical_result_bytes(
    tmp_path: Path,
    grid: Mapping[str, object],
) -> None:
    """Dva běhy nad stejnými vstupy zapíší bajtově shodné agregáty i predikce."""
    inputs = _inputs(tmp_path, grid=grid, visits=VISITS_PER_PATIENT)
    first, second = tmp_path / "first", tmp_path / "second"
    assert _run(inputs, first) is ExitCode.OK
    assert _run(inputs, second) is ExitCode.OK
    assert (first / "classical.parquet").read_bytes() == (
        second / "classical.parquet"
    ).read_bytes()
    assert (first / "classical_predictions.parquet").read_bytes() == (
        second / "classical_predictions.parquet"
    ).read_bytes()


def test_lsa_components_changes_screening_result_bytes(tmp_path: Path) -> None:
    """Dvě a tři komponenty LSA mění výsledky LightGBM nad syntetickým korpusem."""
    random = Random(31)
    vocabulary = [f"word{i}" for i in range(300)]
    corpus = tuple(
        replace(
            note,
            text=" ".join([
                *random.sample(vocabulary, 20),
                "positive" if note.has_label(Diagnosis.DIABETES) else "negative",
            ]),
        )
        for note in notes(visits=VISITS_PER_PATIENT)
    )
    results = []
    for components in (2, 3):
        root = tmp_path / str(components)
        inputs = _inputs(
            root,
            grid=CLASSICAL_GRID | {"families": ("lightgbm_lsa",)},
            visits=VISITS_PER_PATIENT,
        )
        ParquetNoteTable(inputs.cohort).write(corpus)
        config = json.loads(inputs.config.read_text(encoding="utf-8"))
        config["lsa_components"] = components
        inputs.config.write_text(json.dumps(config), encoding="utf-8")
        out = root / "out"
        assert _run(inputs, out) is ExitCode.OK
        results.append((out / "classical.parquet").read_bytes())

    assert results[0] != results[1]


@pytest.mark.parametrize("preprocessing", ["normalized", "dictionary"])
def test_manifest_preserves_preprocessing_and_strip(
    tmp_path: Path, preprocessing: str
) -> None:
    """Manifest nese použitý strip, slovník a explicitní úroveň předzpracování."""
    inputs = _inputs(tmp_path, preprocessing=preprocessing, strip=True)
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.OK
    manifest = ScreeningManifest.model_validate_json(
        (out / "classical_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.text_preprocessing.value == preprocessing
    assert manifest.text_normalization is not None
    assert (
        manifest.text_normalization.acronyms["COPD"]
        == "chronic obstructive pulmonary disease"
    )
    assert bool(manifest.text_normalization.variants) == (preprocessing == "dictionary")
    assert manifest.diagnostic_strip is not None
    assert (
        manifest.diagnostic_strip.lexicon_sha256
        == hashlib.sha256(REPOSITORY_SECTION_HEADERS.read_bytes()).hexdigest()
    )


def test_invalid_revision_refuses_before_any_output(tmp_path: Path) -> None:
    """Nečitelná provenance zabrání výpočtu i zápisu výsledků."""
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(inputs, out, revision=missing_revision) is ExitCode.REFUSED
    assert not out.exists()


def test_main_parses_required_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI předá všechny explicitní cesty a návratový kód run."""
    names = (
        "cohort",
        "split",
        "config",
        "candidates",
        "lexicon",
        "out",
        "predictions",
        "manifest",
        "checkpoint",
        "embeddings",
        "scratch",
    )
    paths = {name: tmp_path / name for name in names}
    monkeypatch.setattr(
        "sys.argv",
        [
            "mediparse-screening",
            *(
                argument
                for name, path in paths.items()
                for argument in (f"--{name}", str(path))
            ),
        ],
    )
    execute = Mock(return_value=ExitCode.REFUSED)
    monkeypatch.setattr(screening, "run", execute)
    monkeypatch.setattr(screening, "configure_logging", Mock())
    assert screening.main() is ExitCode.REFUSED
    assert execute.call_args.kwargs["paths"] == screening.ScreeningPaths(
        corpus=ModelCorpusPaths(
            cohort=paths["cohort"],
            split=paths["split"],
            candidates=paths["candidates"],
            lexicon=paths["lexicon"],
        ),
        config=paths["config"],
        out=paths["out"],
        predictions=paths["predictions"],
        manifest=paths["manifest"],
        checkpoint=paths["checkpoint"],
        embeddings=paths["embeddings"],
        scratch=paths["scratch"],
    )


def test_preprocessing_changes_screening_result_bytes(tmp_path: Path) -> None:
    """Zkratka COPD změní příznaky a agregované výsledky při dictionary oproti none."""
    corpus = tuple(
        replace(
            note,
            text=(
                "COPD wheeze"
                if note.has_label(Diagnosis.DIABETES)
                else "chronic obstructive pulmonary disease cough"
            ),
        )
        for note in notes()
    )
    results = []
    for preprocessing in ("none", "dictionary"):
        root = tmp_path / preprocessing
        inputs = _inputs(root, preprocessing=preprocessing)
        ParquetNoteTable(inputs.cohort).write(corpus)
        out = root / "out"
        assert _run(inputs, out) is ExitCode.OK
        results.append((out / "classical.parquet").read_bytes())

    assert results[0] != results[1]


@pytest.fixture(scope="module")
def screened(tmp_path_factory: pytest.TempPathFactory) -> tuple[_Inputs, Path]:
    """Spustí skutečný screening nad syntetickou kohortou.

    Returns:
        Vstupní soubory a adresář obou výstupů.
    """
    # Dva foldy by při záměně testu za trénink zachovaly celkové pokrytí identit.
    root = tmp_path_factory.mktemp("screened")
    inputs = _inputs(root, visits=4)
    config = json.loads(inputs.config.read_text(encoding="utf-8"))
    config["folds"] = 3
    inputs.config.write_text(json.dumps(config), encoding="utf-8")
    out = root / "out"
    assert _run(inputs, out) is ExitCode.OK
    return inputs, out


def _cell_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[name] for name in ("fold", "vectorizer", "family", "diagnosis"))


def test_predictions_cover_development_notes_once_per_grid_combination(
    screened: tuple[_Inputs, Path],
) -> None:
    """Každá kombinace má přes foldy každou vývojovou zprávu právě jednou."""
    inputs, out = screened
    expected = {
        note.note_id: 1
        for note in ParquetNoteTable(inputs.cohort).read()
        if note.subject_id in DEVELOPMENT
    }
    grid = json.loads(inputs.config.read_text(encoding="utf-8"))
    combinations = set(
        product(
            (spec["name"] for spec in grid["vectorizers"]),
            grid["families"],
            (diagnosis.value for diagnosis in Diagnosis),
        )
    )
    rows = pq.read_table(out / "classical_predictions.parquet").to_pylist()
    assert {
        tuple(row[name] for name in ("vectorizer", "family", "diagnosis"))
        for row in rows
    } == combinations
    for vectorizer, family, diagnosis in combinations:
        assert (
            Counter(
                row["note_id"]
                for row in rows
                if (row["vectorizer"], row["family"], row["diagnosis"])
                == (vectorizer, family, diagnosis)
            )
            == expected
        )


def test_prediction_labels_match_cohort(screened: tuple[_Inputs, Path]) -> None:
    """Pravdivý label řádku patří dané diagnóze a identitě zprávy v kohortě."""
    inputs, out = screened
    cohort = {note.note_id: note for note in ParquetNoteTable(inputs.cohort).read()}
    rows = pq.read_table(out / "classical_predictions.parquet").to_pylist()
    assert rows
    for row in rows:
        assert row["label"] == cohort[row["note_id"]].has_label(
            Diagnosis(row["diagnosis"])
        )


def test_metrics_recompute_from_predictions(screened: tuple[_Inputs, Path]) -> None:
    """Ruční TP/FP/FN/TN reprodukují metriky a počty každé buňky agregátů."""
    _, out = screened
    predictions = pq.read_table(out / "classical_predictions.parquet").to_pylist()
    for result in pq.read_table(out / "classical.parquet").to_pylist():
        rows = [row for row in predictions if _cell_key(row) == _cell_key(result)]
        counts = Counter((row["label"], row["predicted"]) for row in rows)
        tp, fp = counts[True, True], counts[False, True]
        fn, tn = counts[True, False], counts[False, False]
        assert result["test_notes"] == len(rows)
        assert result["test_positives"] == sum(row["label"] for row in rows)
        assert result["precision"] == pytest.approx(tp / (tp + fp) if tp + fp else 0.0)
        assert result["recall"] == pytest.approx(tp / (tp + fn) if tp + fn else 0.0)
        assert result["f1"] == pytest.approx(
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        )
        assert result["accuracy"] == pytest.approx((tp + tn) / (tp + fp + fn + tn))


def test_predictions_use_cell_threshold(screened: tuple[_Inputs, Path]) -> None:
    """Binární predikce respektuje skóre a práh odpovídající buňky."""
    _, out = screened
    thresholds = {
        _cell_key(row): row["threshold"]
        for row in pq.read_table(out / "classical.parquet").to_pylist()
    }
    rows = pq.read_table(out / "classical_predictions.parquet").to_pylist()
    assert rows
    for row in rows:
        assert row["predicted"] == (row["score"] >= thresholds[_cell_key(row)])


@pytest.mark.parametrize(
    "changed",
    ["config", "split", "candidates", "lexicon", "dataset_sha256", "packages"],
)
def test_checkpoint_key_changes_with_each_input(tmp_path: Path, changed: str) -> None:
    """Každý vědecký vstup a verze knihoven jsou součástí klíče."""
    inputs = _inputs(tmp_path)
    corpus = repository_corpus(inputs.cohort, inputs.split)
    for field in ("candidates", "lexicon"):
        copy = tmp_path / field
        copy.write_bytes(getattr(corpus, field).read_bytes())
        corpus = replace(corpus, **{field: copy})
    paths = screening.ScreeningPaths(
        corpus=corpus,
        config=inputs.config,
        out=tmp_path / "out",
        predictions=tmp_path / "predictions",
        manifest=tmp_path / "manifest",
        checkpoint=tmp_path / "checkpoint",
        embeddings=tmp_path / "embeddings",
        scratch=tmp_path / "scratch",
    )
    packages = {"python": "test"}
    before = screening.checkpoint_key(paths.config, paths.corpus, "dataset", packages)
    assert (
        screening.checkpoint_key(paths.config, paths.corpus, "dataset", packages)
        == before
    )
    dataset = "dataset"
    if changed == "dataset_sha256":
        dataset = "other"
    elif changed == "packages":
        packages = {"python": "other"}
    else:
        path = paths.config if changed == "config" else getattr(paths.corpus, changed)
        path.write_bytes(path.read_bytes() + b" ")
    assert (
        screening.checkpoint_key(paths.config, paths.corpus, dataset, packages)
        != before
    )


def test_foreign_checkpoint_refuses_before_outputs(tmp_path: Path) -> None:
    """Cizí checkpoint odmítne běh před výsledky, predikcemi i manifestem."""
    inputs = _inputs(tmp_path)
    out = tmp_path / "out"
    checkpoint = out / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "key.json").write_text('{"key":"other"}', encoding="utf-8")
    assert _run(inputs, out) is ExitCode.REFUSED
    assert {p.name for p in out.iterdir()} == {"checkpoint"}


def test_encoder_screening_end_to_end(tmp_path: Path) -> None:
    """Malý encoder dodá příznaky LR, cache i úplnou provenance bez stahování."""
    pytest.importorskip("torch")

    inputs = _inputs(tmp_path, grid=CLASSICAL_GRID | {"families": ("logreg",)})
    spec = SPEC.model_copy(update={"name": "mmbert-base-mean"})
    config = json.loads(inputs.config.read_text(encoding="utf-8"))
    config["vectorizers"] = [spec.model_dump(mode="json")]
    inputs.config.write_text(json.dumps(config), encoding="utf-8")
    model, tokenizer = encoder_parts(
        sorted({word for note in notes() for word in note.text.split()})
    )
    factory = Mock(
        return_value=HfMeanEncoder(model, tokenizer, spec.window, spec.device)
    )
    out = tmp_path / "out"
    paths = screening.ScreeningPaths(
        corpus=repository_corpus(inputs.cohort, inputs.split),
        config=inputs.config,
        out=out / "results.parquet",
        predictions=out / "predictions.parquet",
        manifest=out / "manifest.json",
        checkpoint=out / "checkpoint",
        embeddings=out / "embeddings",
        scratch=out / "scratch",
    )
    assert (
        screening.run(
            environ={}, paths=paths, revision=fixed_revision, encoders=factory
        )
        is ExitCode.OK
    )
    rows = pq.read_table(paths.out).to_pylist()
    assert len(rows) == config["folds"] * len(Diagnosis)
    assert {row["vectorizer"] for row in rows} == {"mmbert-base-mean"}
    assert {row["family"] for row in rows} == {"logreg"}
    manifest = ScreeningManifest.model_validate_json(paths.manifest.read_bytes())
    assert manifest.encoders == (spec,)
    assert set(ENCODER_PACKAGES) <= manifest.environment.packages.keys()
    assert list(paths.embeddings.glob("*/*.parquet"))
    assert not paths.checkpoint.exists()


def test_encoder_revision_and_versions_change_checkpoint_key(tmp_path: Path) -> None:
    """Revize i knihovny encoderu mění obnovu také v prostředí bez torch."""
    inputs = _inputs(tmp_path, grid=CLASSICAL_GRID | {"families": ("logreg",)})
    config = json.loads(inputs.config.read_text(encoding="utf-8"))
    config["vectorizers"] = [SPEC.model_dump(mode="json")]
    inputs.config.write_text(json.dumps(config), encoding="utf-8")
    paths = screening.ScreeningPaths(
        corpus=repository_corpus(inputs.cohort, inputs.split),
        config=inputs.config,
        out=tmp_path / "results",
        predictions=tmp_path / "predictions",
        manifest=tmp_path / "manifest",
        checkpoint=tmp_path / "checkpoint",
        embeddings=tmp_path / "embeddings",
        scratch=tmp_path / "scratch",
    )
    before = screening.checkpoint_key(paths.config, paths.corpus, "dataset", VERSIONS)
    assert (
        screening.checkpoint_key(
            paths.config, paths.corpus, "dataset", VERSIONS | {"torch": "other"}
        )
        != before
    )
    config["vectorizers"][0]["revision"] = "b" * 40
    inputs.config.write_text(json.dumps(config), encoding="utf-8")
    assert (
        screening.checkpoint_key(paths.config, paths.corpus, "dataset", VERSIONS)
        != before
    )


def test_fasttext_screening_end_to_end(tmp_path: Path) -> None:
    """Tokeny projdou skutečným screeningem s provenance a úklidem checkpointu."""
    inputs = _inputs(
        tmp_path,
        grid={"families": ("fasttext",), "vectorizers": [{"name": "word-tokens"}]},
        visits=VISITS_PER_PATIENT,
    )
    out = tmp_path / "out"
    assert _run(inputs, out) is ExitCode.OK
    rows = pq.read_table(out / "classical.parquet").to_pylist()
    config = json.loads(inputs.config.read_text(encoding="utf-8"))
    assert len(rows) == config["folds"] * len(Diagnosis)
    assert {row["family"] for row in rows} == {"fasttext"}
    assert {row["vectorizer"] for row in rows} == {"word-tokens"}
    manifest = ScreeningManifest.model_validate_json(
        (out / "classical_manifest.json").read_bytes()
    )
    assert manifest.environment.packages["fasttext"] == "0.9.2"
    assert not (out / "checkpoint").exists()
    assert list((out / "scratch").iterdir()) == []
