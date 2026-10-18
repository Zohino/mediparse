"""Vstupní bod kroku evaluate_smoketest_model: metriky z predikcí a odmítnutí vstupu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from mediparse.domain.diagnosis import Diagnosis
from mediparse.entrypoints import smoketest_evaluation
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from tests.model_support import prediction

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_writes_metrics_row(tmp_path: Path) -> None:
    """Predikce dají metrics.parquet s jedním řádkem diagnózy a hodnotami v [0, 1]."""
    predictions = tmp_path / "predictions.parquet"
    metrics = tmp_path / "build" / "metrics.parquet"
    ParquetPredictionTable(predictions).write([
        prediction(
            "1-DS-1",
            truth=True,
            score=2.0,
            diagnosis=Diagnosis.DIABETES,
            row_id="row",
        ),
        prediction(
            "2-DS-1",
            truth=True,
            score=-0.5,
            diagnosis=Diagnosis.DIABETES,
            row_id="row",
        ),
        prediction(
            "3-DS-1",
            truth=False,
            score=-1.0,
            diagnosis=Diagnosis.DIABETES,
            row_id="row",
        ),
        prediction(
            "4-DS-1",
            truth=False,
            score=0.5,
            diagnosis=Diagnosis.DIABETES,
            row_id="row",
        ),
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
        prediction(
            "1-DS-1",
            truth=True,
            score=1.0,
            diagnosis=Diagnosis.DIABETES,
            row_id="row",
        ),
        prediction(
            "2-DS-1",
            truth=True,
            score=2.0,
            diagnosis=Diagnosis.DIABETES,
            row_id="row",
        ),
    ])

    code = smoketest_evaluation.run(predictions=predictions, metrics=metrics)

    assert code is ExitCode.REFUSED
    assert not metrics.exists()
    assert "diabetes" in caplog.text
    assert "Traceback" not in caplog.text
