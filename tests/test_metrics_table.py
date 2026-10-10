"""Parquet adaptér metrik: pevné schéma a hodnoty z DiagnosisMetrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.metrics_table import SCHEMA, ParquetMetricsTable

if TYPE_CHECKING:
    from pathlib import Path


def test_written_file_has_schema_and_values(tmp_path: Path) -> None:
    """Zapsaný soubor má přesně schéma a hodnoty, nadřazený adresář se založí."""
    path = tmp_path / "out" / "metrics.parquet"
    metrics = [
        DiagnosisMetrics(
            "row",
            Diagnosis.CKD,
            40,
            12,
            BinaryMetrics(0.5, 0.25, 0.75, 0.875, 0.9, 0.8),
        )
    ]

    ParquetMetricsTable(path).write(metrics)

    table = pq.read_table(path)
    assert table.schema.equals(SCHEMA)
    assert table.to_pylist() == [
        {
            "row_id": "row",
            "diagnosis": "ckd",
            "test_notes": 40,
            "test_positives": 12,
            "precision": 0.5,
            "recall": 0.25,
            "f1": 0.75,
            "accuracy": 0.875,
            "roc_auc": 0.9,
            "pr_auc": 0.8,
        }
    ]
