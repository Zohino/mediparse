"""Vstupní bod kroku evaluate_smoketest_model: metriky z predikcí a odmítnutí vstupu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from mediparse.domain.evaluation import Prediction
from mediparse.domain.labels import Diagnosis
from mediparse.entrypoints import smoketest_evaluation
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.prediction_table import ParquetPredictionTable

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _prediction(note: str, *, truth: bool, score: float) -> Prediction:
    return Prediction(
        "row",
        0,
        note,
        1,
        Diagnosis.DIABETES,
        truth=truth,
        predicted=score > 0,
        score=score,
    )


def test_writes_metrics_row(tmp_path: Path) -> None:
    """Predikce dají metrics.parquet s jedním řádkem diagnózy a hodnotami v [0, 1]."""
    predictions = tmp_path / "predictions.parquet"
    metrics = tmp_path / "build" / "metrics.parquet"
    ParquetPredictionTable(predictions).write([
        _prediction("a", truth=True, score=2.0),
        _prediction("b", truth=True, score=-0.5),
        _prediction("c", truth=False, score=-1.0),
        _prediction("d", truth=False, score=0.5),
    ])

    code = smoketest_evaluation.run(predictions=predictions, metrics=metrics)

    rows = pq.read_table(metrics).to_pylist()
    assert code is ExitCode.OK
    assert len(rows) == 1
    assert (rows[0]["row_id"], rows[0]["diagnosis"]) == ("row", "diabetes")
    assert (rows[0]["test_notes"], rows[0]["test_positives"]) == (4, 2)
    for name in ("precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"):
        assert 0.0 <= rows[0][name] <= 1.0


def test_missing_predictions_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící soubor predikcí krok odmítne a nic nevznikne."""
    metrics = tmp_path / "metrics.parquet"

    code = smoketest_evaluation.run(
        predictions=tmp_path / "none.parquet", metrics=metrics
    )

    assert code is ExitCode.REFUSED
    assert not metrics.exists()
    assert "none.parquet" in caplog.text


def test_single_class_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Jediná třída pravdy krok odmítne bez tracebacku a nic nevznikne."""
    predictions = tmp_path / "predictions.parquet"
    metrics = tmp_path / "metrics.parquet"
    ParquetPredictionTable(predictions).write([
        _prediction("a", truth=True, score=1.0),
        _prediction("b", truth=True, score=2.0),
    ])

    code = smoketest_evaluation.run(predictions=predictions, metrics=metrics)

    assert code is ExitCode.REFUSED
    assert not metrics.exists()
    assert "diabetes" in caplog.text
    assert "Traceback" not in caplog.text
