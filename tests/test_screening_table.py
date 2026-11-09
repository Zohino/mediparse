"""Pevné schéma a zachování pořadí agregované tabulky screeningu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.classical_screening import ClassifierFamily, ScreeningResult
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics
from mediparse.infrastructure.screening_table import SCHEMA, ParquetScreeningTable

if TYPE_CHECKING:
    from pathlib import Path


def test_written_schema_and_rows_match_contract(tmp_path: Path) -> None:
    """Parquet zachová pořadí a hodnoty vstupů i přesné nenulovatelné typy kontraktu."""
    path = tmp_path / "out" / "screening.parquet"
    rows = [
        ScreeningResult(
            fold=fold,
            vectorizer="word",
            family=ClassifierFamily.LOGREG,
            diagnosis=Diagnosis.CKD,
            test_notes=40,
            test_positives=12,
            threshold=0.25,
            converged=converged,
            metrics=BinaryMetrics(0.5, 0.25, 0.75, 0.875, 0.9, 0.8),
        )
        for fold, converged in [(1, False), (0, True)]
    ]
    expected_schema = pa.schema([
        pa.field(name, kind, nullable=False)
        for name, kind in [
            ("fold", pa.int64()),
            ("vectorizer", pa.string()),
            ("family", pa.string()),
            ("diagnosis", pa.string()),
            ("test_notes", pa.int64()),
            ("test_positives", pa.int64()),
            ("threshold", pa.float64()),
            ("converged", pa.bool_()),
            ("precision", pa.float64()),
            ("recall", pa.float64()),
            ("f1", pa.float64()),
            ("accuracy", pa.float64()),
            ("roc_auc", pa.float64()),
            ("pr_auc", pa.float64()),
        ]
    ])
    ParquetScreeningTable(path).write(rows)
    table = pq.read_table(path)
    assert table.schema == expected_schema == SCHEMA
    assert table.to_pylist() == [
        {
            "fold": fold,
            "vectorizer": "word",
            "family": "logreg",
            "diagnosis": "ckd",
            "test_notes": 40,
            "test_positives": 12,
            "threshold": 0.25,
            "converged": converged,
            "precision": 0.5,
            "recall": 0.25,
            "f1": 0.75,
            "accuracy": 0.875,
            "roc_auc": 0.9,
            "pr_auc": 0.8,
        }
        for fold, converged in [(1, False), (0, True)]
    ]
