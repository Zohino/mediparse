"""Vstupní bod kroku smoketest: souhrn metrik na výstup a odmítnutí chybějících metrik."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics
from mediparse.entrypoints import smoketest_summary
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.metrics_table import ParquetMetricsTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from tests.model_support import prediction

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _inference(root: Path) -> Path:
    inference = root / "inference.parquet"
    ParquetPredictionTable(inference).write([
        prediction("9002-DS-1", truth=True, score=2.0, diagnosis=Diagnosis.DIABETES),
        prediction(
            "9000-DS-1",
            truth=True,
            score=1.23456,
            diagnosis=Diagnosis.DIABETES,
        ),
        prediction(
            "9001-DS-1",
            truth=False,
            score=0.1,
            diagnosis=Diagnosis.DIABETES,
        ),
    ])
    return inference


def _run(
    root: Path,
    out: io.StringIO,
    *,
    metrics: Path | None = None,
    inference: Path | None = None,
) -> ExitCode:
    return smoketest_summary.run(
        metrics=metrics or _metrics(root),
        inference=inference or _inference(root),
        corpus=root / "korpus",
        outputs=root / "vystupy",
        logs=root / "logy",
        out=out,
    )


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

    code = _run(tmp_path, out, metrics=metrics)

    lines = out.getvalue().splitlines()
    diabetes = next(line for line in lines if "diabetes" in line).split()
    ckd = next(line for line in lines if "ckd" in line).split()
    assert code is ExitCode.OK
    assert diabetes == ["row", "diabetes", "40", "12", "0.833", "0.917", "0.750"]
    assert ckd == ["row", "ckd", "38", "9", "0.450", "0.700", "0.600"]
    assert f"Data, model, predikce, metriky a manifest: {tmp_path / 'vystupy'}/\n" in (
        out.getvalue()
    )
    assert f"Logy kroků: {tmp_path / 'logy'}/\n" in out.getvalue()


def test_missing_metrics_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící metriky krok odmítnou bez tracebacku a nic se nevypíše."""
    out = io.StringIO()

    code = _run(tmp_path, out, metrics=tmp_path / "none.parquet")

    assert code is ExitCode.REFUSED
    assert not out.getvalue()
    assert "none.parquet" in caplog.text
    assert "Traceback" not in caplog.text


def test_summary_selects_first_positive_and_negative(
    tmp_path: Path,
) -> None:
    """Ze tří predikcí vybere první pozitivní a první negativní podle note_id."""
    out = io.StringIO()

    code = _run(tmp_path, out)

    text = out.getvalue()
    lines = text.splitlines()
    first = next(line for line in lines if "9000-DS-1" in line).split()
    second = next(line for line in lines if "9001-DS-1" in line).split()
    assert code is ExitCode.OK
    assert text.index("ROC-AUC") < text.index("9000-DS-1")
    assert first == [
        str(tmp_path / "korpus" / "en" / "9000-DS-1.txt"),
        "diabetes",
        "ano",
        "ano",
        "1.235",
    ]
    assert second[2:] == ["ne", "ano", "0.100"]
    demo_lines = [line for line in lines if "-DS-" in line]
    assert [line.split()[0] for line in demo_lines] == [
        str(tmp_path / "korpus" / "en" / "9000-DS-1.txt"),
        str(tmp_path / "korpus" / "en" / "9001-DS-1.txt"),
    ]
    assert "kladné skóre znamená ano" in text


def test_missing_inference_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící inference.parquet krok odmítne bez tracebacku a nic se nevypíše."""
    out = io.StringIO()

    code = _run(tmp_path, out, inference=tmp_path / "none.parquet")

    assert code is ExitCode.REFUSED
    assert not out.getvalue()
    assert "none.parquet" in caplog.text
    assert "Traceback" not in caplog.text
