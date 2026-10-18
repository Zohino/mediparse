"""Vstupní bod kroku evaluate_smoketest_model: metriky z predikcí a odmítnutí vstupu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow.parquet as pq

from mediparse.domain.diagnosis import Diagnosis
from mediparse.entrypoints import smoketest_evaluation
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from tests.model_support import prediction, separable_notes, write_split

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest

    from mediparse.domain.evaluation import Prediction

PATIENTS: Final = 12


@dataclass(frozen=True)
class _Inputs:
    predictions: Path
    notes: Path
    split: Path
    metrics: Path
    test_predictions: tuple[Prediction, ...]

    def run(self) -> ExitCode:
        return smoketest_evaluation.run(
            predictions=self.predictions,
            notes=self.notes,
            split=self.split,
            metrics=self.metrics,
        )


def _inputs(root: Path) -> _Inputs:
    corpus = separable_notes(PATIENTS)
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(corpus)
    split = root / "splits.parquet"
    shared = write_split(split, corpus)
    test = shared.split.holdout(corpus, 0).test
    items = tuple(
        prediction(
            note.note_id,
            truth=Diagnosis.DIABETES in note.labels,
            score=1.0 if Diagnosis.DIABETES in note.labels else -1.0,
        )
        for note in test
    )
    predictions = root / "predictions.parquet"
    ParquetPredictionTable(predictions).write(items)
    return _Inputs(predictions, notes, split, root / "build" / "metrics.parquet", items)


def _rewrite_predictions(inputs: _Inputs, items: Sequence[Prediction]) -> None:
    ParquetPredictionTable(inputs.predictions).write(items)


def test_writes_metrics_row(tmp_path: Path) -> None:
    """Predikce testu foldu 0 dají metrics.parquet s jedním řádkem a hodnotami v [0, 1]."""
    inputs = _inputs(tmp_path)

    code = inputs.run()

    rows = pq.read_table(inputs.metrics).to_pylist()
    assert code is ExitCode.OK
    assert len(rows) == 1
    assert (rows[0]["row_id"], rows[0]["diagnosis"]) == ("row", "diabetes")
    assert rows[0]["test_notes"] == len(inputs.test_predictions)
    assert rows[0]["test_positives"] == sum(
        item.truth for item in inputs.test_predictions
    )
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
    foreign = prediction("2-DS-1", truth=True, score=1.0, fold=0)
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
    for name in ("predictions", "notes", "split"):
        root = tmp_path / name
        root.mkdir()
        inputs = _inputs(root)
        missing = root / "none.parquet"
        paths = {
            "predictions": inputs.predictions,
            "notes": inputs.notes,
            "split": inputs.split,
        } | {name: missing}

        code = smoketest_evaluation.run(metrics=inputs.metrics, **paths)

        assert code is ExitCode.REFUSED
        assert not inputs.metrics.exists()
    assert caplog.text.count("none.parquet") == len(("predictions", "notes", "split"))


def test_single_class_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Jediná třída pravdy krok odmítne bez tracebacku a nic nevznikne."""
    inputs = _inputs(tmp_path)
    _rewrite_predictions(
        inputs,
        [
            prediction(item.note_id, truth=True, score=item.score)
            for item in inputs.test_predictions
        ],
    )

    code = inputs.run()

    assert code is ExitCode.REFUSED
    assert not inputs.metrics.exists()
    assert "diabetes" in caplog.text
    assert "Traceback" not in caplog.text
