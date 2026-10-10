"""Vstupní bod kroku smoketest: souhrn metrik na výstup a odmítnutí chybějících metrik."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics, Prediction
from mediparse.domain.labels import Diagnosis
from mediparse.entrypoints import smoketest_summary
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.metrics_table import ParquetMetricsTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _demo(root: Path) -> Path:
    demo = root / "demo.parquet"
    ParquetPredictionTable(demo).write([
        Prediction(
            "row",
            0,
            "9000-DS-1",
            1,
            Diagnosis.DIABETES,
            truth=True,
            predicted=True,
            score=1.23456,
        ),
        Prediction(
            "row",
            0,
            "9001-DS-1",
            2,
            Diagnosis.DIABETES,
            truth=False,
            predicted=True,
            score=0.1,
        ),
    ])
    return demo


def _metrics(root: Path) -> Path:
    metrics = root / "metrics.parquet"
    ParquetMetricsTable(metrics).write([
        DiagnosisMetrics(
            "row", Diagnosis.CKD, 38, 9, BinaryMetrics(0.6, 0.35, 0.45, 0.5, 0.7, 0.6)
        )
    ])
    return metrics


def test_summary_lists_each_diagnosis_and_output_paths(tmp_path: Path) -> None:
    """Souhrn má řádek na diagnózu s metrikami na 3 desetinná místa a cesty k výstupům."""
    metrics = tmp_path / "metrics.parquet"
    ParquetMetricsTable(metrics).write([
        DiagnosisMetrics(
            "row",
            Diagnosis.DIABETES,
            40,
            12,
            BinaryMetrics(0.5, 0.25, 0.8333333, 0.875, 0.9166667, 0.75),
        ),
        DiagnosisMetrics(
            "row", Diagnosis.CKD, 38, 9, BinaryMetrics(0.6, 0.35, 0.45, 0.5, 0.7, 0.6)
        ),
    ])
    out = io.StringIO()

    code = smoketest_summary.run(metrics=metrics, demo=_demo(tmp_path), out=out)

    lines = out.getvalue().splitlines()
    diabetes = next(line for line in lines if "diabetes" in line).split()
    ckd = next(line for line in lines if "ckd" in line).split()
    assert code is ExitCode.OK
    assert diabetes == ["row", "diabetes", "40", "12", "0.833", "0.917", "0.750"]
    assert ckd == ["row", "ckd", "38", "9", "0.450", "0.700", "0.600"]
    assert "build/smoketest/" in out.getvalue()
    assert "logs/smoketest/" in out.getvalue()


def test_missing_metrics_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící metriky krok odmítnou bez tracebacku a nic se nevypíše."""
    out = io.StringIO()

    code = smoketest_summary.run(
        metrics=tmp_path / "none.parquet", demo=_demo(tmp_path), out=out
    )

    assert code is ExitCode.REFUSED
    assert not out.getvalue()
    assert "none.parquet" in caplog.text
    assert "Traceback" not in caplog.text


def test_summary_shows_demo_with_path_truth_prediction_and_score(
    tmp_path: Path,
) -> None:
    """Pod metrikami je ukázka: cesta ke zprávě, diagnóza, pravda, predikce a skóre."""
    out = io.StringIO()

    code = smoketest_summary.run(
        metrics=_metrics(tmp_path), demo=_demo(tmp_path), out=out
    )

    text = out.getvalue()
    lines = text.splitlines()
    first = next(line for line in lines if "9000-DS-1" in line).split()
    second = next(line for line in lines if "9001-DS-1" in line).split()
    assert code is ExitCode.OK
    assert text.index("ROC-AUC") < text.index("9000-DS-1")
    assert first == [
        "resources/synthetic/en/9000-DS-1.txt",
        "diabetes",
        "ano",
        "ano",
        "1.235",
    ]
    assert second[2:] == ["ne", "ano", "0.100"]
    assert "kladné skóre znamená ano" in text


def test_missing_demo_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící demo.parquet krok odmítne bez tracebacku a nic se nevypíše."""
    out = io.StringIO()

    code = smoketest_summary.run(
        metrics=_metrics(tmp_path), demo=tmp_path / "none.parquet", out=out
    )

    assert code is ExitCode.REFUSED
    assert not out.getvalue()
    assert "none.parquet" in caplog.text
    assert "Traceback" not in caplog.text
