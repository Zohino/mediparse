"""Zápis reportu tréninku do JSON."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_training import BinaryMetrics, TrainingReport
from mediparse.infrastructure.report_file import JsonReportFile

if TYPE_CHECKING:
    from pathlib import Path


def test_writes_flat_sorted_report(tmp_path: Path) -> None:
    """Report je plochý JSON se seřazenými klíči; adresář se založí."""
    path = tmp_path / "build" / "metrics.json"
    report = TrainingReport(
        Diagnosis.DIABETES, 160, 40, 12, BinaryMetrics(0.5, 0.25, 0.75, 0.875)
    )

    JsonReportFile(path).write(report)

    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == {
        "diagnosis": "diabetes",
        "train_notes": 160,
        "test_notes": 40,
        "test_positives": 12,
        "precision": 0.5,
        "recall": 0.25,
        "f1": 0.75,
        "accuracy": 0.875,
    }
    assert list(json.loads(text)) == sorted(json.loads(text))
