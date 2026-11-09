"""Vstupní bod kroku evaluate_model: metriky z predikcí a odmítnutí vstupu."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.run_manifest import EvaluationManifest
from mediparse.domain.split import SharedSplit, Split
from mediparse.entrypoints import model_evaluation
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.manifest_support import training_manifest
from tests.model_support import prediction, separable_notes, split_config, write_split
from tests.support import REVISION, fixed_revision, missing_revision

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.evaluation import Prediction

PATIENTS: Final = 12


@dataclass(frozen=True)
class _Inputs:
    predictions: Path
    notes: Path
    split: Path
    metrics: Path
    training_manifest: Path
    manifest: Path
    test_predictions: tuple[Prediction, ...]

    def run(self) -> ExitCode:
        return model_evaluation.run(
            environ={},
            predictions=self.predictions,
            cohort=self.notes,
            split=self.split,
            metrics=self.metrics,
            training_manifest=self.training_manifest,
            manifest=self.manifest,
            revision=fixed_revision,
        )


def _inputs(root: Path, shared: SharedSplit | None = None) -> _Inputs:
    corpus = separable_notes(PATIENTS)
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(corpus)
    split = root / "splits.parquet"
    if shared is None:
        shared = write_split(split, corpus)
    else:
        ParquetSplitTable(split).write(shared)
    test = shared.split.holdout(corpus, 0).test
    items = tuple(
        prediction(
            note.note_id,
            score=1.0 if Diagnosis.DIABETES in note.labels else -1.0,
        )
        for note in test
    )
    predictions = root / "predictions.parquet"
    ParquetPredictionTable(predictions).write(items)
    inputs = _Inputs(
        predictions,
        notes,
        split,
        root / "build" / "metrics.parquet",
        root / "run_manifest.json",
        root / "build" / "evaluation_manifest.json",
        items,
    )
    _record_training(inputs)
    return inputs


def _record_training(inputs: _Inputs) -> None:
    write_model(
        inputs.training_manifest,
        training_manifest(
            predictions_sha256=hashlib.sha256(
                inputs.predictions.read_bytes()
            ).hexdigest(),
            dataset_sha256=hashlib.sha256(inputs.notes.read_bytes()).hexdigest(),
            split_sha256=ParquetSplitTable(inputs.split).read().split.sha256(),
        ),
    )


def _rewrite_predictions(inputs: _Inputs, items: Sequence[Prediction]) -> None:
    ParquetPredictionTable(inputs.predictions).write(items)
    _record_training(inputs)


def test_writes_metrics_row(tmp_path: Path) -> None:
    """Predikce testu foldu 0 dají metrics.parquet s jedním řádkem a hodnotami v [0, 1]."""
    inputs = _inputs(tmp_path)

    code = inputs.run()

    rows = pq.read_table(inputs.metrics).to_pylist()
    assert code is ExitCode.OK
    assert len(rows) == 1
    assert (rows[0]["row_id"], rows[0]["diagnosis"]) == ("row", "diabetes")
    assert rows[0]["test_notes"] == len(inputs.test_predictions)
    for name in ("precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"):
        assert 0.0 <= rows[0][name] <= 1.0


def test_incomplete_predictions_are_refused_with_count_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Predikce bez jedné testovací zprávy foldu krok odmítne, log nese počet bez note_id."""
    inputs = _inputs(tmp_path)
    _rewrite_predictions(inputs, inputs.test_predictions[1:])

    code = inputs.run()

    assert code is ExitCode.REFUSED
    assert not inputs.metrics.exists()
    assert "chybí 1, navíc 0" in caplog.text
    assert "-DS-" not in caplog.text


def test_prediction_of_note_outside_test_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Predikce zprávy, která do testu foldu nepatří, krok odmítne."""
    inputs = _inputs(tmp_path)
    foreign = prediction("2-DS-1", score=1.0, fold=0)
    assert foreign.note_id not in {item.note_id for item in inputs.test_predictions}
    _rewrite_predictions(inputs, (*inputs.test_predictions, foreign))

    code = inputs.run()

    assert code is ExitCode.REFUSED
    assert not inputs.metrics.exists()
    assert "navíc 1" in caplog.text


def test_missing_input_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící predikce, zprávy nebo split krok odmítne a log jmenuje soubor."""
    for name in ("predictions", "cohort", "split"):
        root = tmp_path / name
        root.mkdir()
        inputs = _inputs(root)
        missing = root / "none.parquet"
        paths = {
            "predictions": inputs.predictions,
            "cohort": inputs.notes,
            "split": inputs.split,
        } | {name: missing}

        code = model_evaluation.run(
            environ={},
            metrics=inputs.metrics,
            training_manifest=inputs.training_manifest,
            manifest=inputs.manifest,
            revision=fixed_revision,
            **paths,
        )

        assert code is ExitCode.REFUSED
        assert not inputs.metrics.exists()
    assert caplog.text.count("none.parquet") == len(("predictions", "cohort", "split"))


def test_single_class_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Jediná třída pravdy krok odmítne bez tracebacku a nic nevznikne."""
    inputs = _inputs(tmp_path)
    table = ParquetNoteTable(inputs.notes)
    test_ids = {item.note_id for item in inputs.test_predictions}
    table.write([
        replace(note, labels=note.labels | {Diagnosis.DIABETES})
        if note.note_id in test_ids
        else note
        for note in table.read()
    ])

    _record_training(inputs)
    code = inputs.run()

    assert code is ExitCode.REFUSED
    assert not inputs.metrics.exists()
    assert "diabetes" in caplog.text
    assert "metriky potřebují obě třídy" in caplog.text
    assert "Traceback" not in caplog.text


def test_all_negative_predictions_are_evaluated_against_labels(tmp_path: Path) -> None:
    """Samé záporné predikce mají F1 a recall nula a při stejných skóre ROC-AUC půl."""
    shared = SharedSplit(
        split_config(),
        Split(
            (frozenset({1, 4, 9, 11}), frozenset({2, 3, 5, 6, 7, 8})),
            development=frozenset({10, 12}),
        ),
    )
    inputs = _inputs(tmp_path, shared)
    _rewrite_predictions(
        inputs,
        [prediction(item.note_id, score=-1.0) for item in inputs.test_predictions],
    )

    code = inputs.run()

    assert code is ExitCode.OK
    (row,) = pq.read_table(inputs.metrics).to_pylist()
    assert row["f1"] == pytest.approx(0.0)
    assert row["recall"] == pytest.approx(0.0)
    assert row["roc_auc"] == pytest.approx(0.5)
    # Ruční součet z fixtury: pozitivní pacienti 1, 9 a 11 ve foldu 0 mají 1 + 5 + 3 zpráv.
    assert row["test_positives"] == 9  # ruff: ignore[magic-value-comparison]


def test_evaluation_manifest_records_digests_and_is_deterministic(
    tmp_path: Path,
) -> None:
    """A6: manifest váže bajty metrik a všech vstupů a opakování ho nezmění."""
    inputs = _inputs(tmp_path)
    assert inputs.run() is ExitCode.OK
    first = inputs.manifest.read_bytes()
    manifest = EvaluationManifest.model_validate_json(first)

    for field, path in (
        ("metrics_sha256", inputs.metrics),
        ("training_manifest_sha256", inputs.training_manifest),
        ("predictions_sha256", inputs.predictions),
        ("dataset_sha256", inputs.notes),
    ):
        assert getattr(manifest, field) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        manifest.split_sha256
        == json.loads(inputs.training_manifest.read_bytes())["split_sha256"]
    )
    assert manifest.row_id == "row"
    assert manifest.environment.source == REVISION
    assert tuple(manifest.environment.packages) == model_evaluation.PACKAGES
    assert inputs.run() is ExitCode.OK
    assert inputs.manifest.read_bytes() == first


@pytest.mark.parametrize("input_name", ["cohort", "split"])
def test_changed_model_input_table_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, input_name: str
) -> None:
    """A5: jiné bajty kohorty se stejnými klíči i jiné přiřazení splitu se odmítnou."""
    inputs = _inputs(tmp_path)
    table = ParquetNoteTable(inputs.notes)
    if input_name == "cohort":
        notes = table.read()
        table.write([replace(notes[0], text=notes[0].text + " synthetic"), *notes[1:]])
        field = "dataset_sha256"
    else:
        shared = ParquetSplitTable(inputs.split).read()
        ParquetSplitTable(inputs.split).write(
            replace(
                shared,
                split=replace(shared.split, folds=tuple(reversed(shared.split.folds))),
            )
        )
        field = "split_sha256"

    assert inputs.run() is ExitCode.REFUSED
    assert field in caplog.text
    assert str(inputs.training_manifest) in caplog.text
    assert not inputs.metrics.exists()
    assert not inputs.manifest.exists()


@pytest.mark.parametrize("content", [None, "{}", "neplatný JSON"])
def test_missing_or_invalid_training_manifest_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str | None
) -> None:
    """Chybějící nebo neplatný manifest skončí REFUSED před zápisem výstupů."""
    inputs = _inputs(tmp_path)
    inputs.training_manifest.unlink()
    if content is not None:
        inputs.training_manifest.write_text(content, encoding="utf-8")

    assert inputs.run() is ExitCode.REFUSED
    assert str(inputs.training_manifest) in caplog.text
    assert not inputs.metrics.exists()
    assert not inputs.manifest.exists()


def test_unreadable_revision_is_refused_before_evaluation(tmp_path: Path) -> None:
    """Nečitelná revize odmítne evaluaci před prvním výstupem."""
    inputs = _inputs(tmp_path)
    code = model_evaluation.run(
        environ={},
        predictions=inputs.predictions,
        cohort=inputs.notes,
        split=inputs.split,
        training_manifest=inputs.training_manifest,
        metrics=inputs.metrics,
        manifest=inputs.manifest,
        revision=missing_revision,
    )
    assert code is ExitCode.REFUSED
    assert not inputs.metrics.exists()
    assert not inputs.manifest.exists()
