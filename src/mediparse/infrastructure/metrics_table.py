"""Metriky evaluace v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.evaluation import DiagnosisMetrics

SCHEMA: Final = pa.schema([
    pa.field("row_id", pa.string(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
    pa.field("test_notes", pa.int64(), nullable=False),
    pa.field("test_positives", pa.int64(), nullable=False),
    pa.field("precision", pa.float64(), nullable=False),
    pa.field("recall", pa.float64(), nullable=False),
    pa.field("f1", pa.float64(), nullable=False),
    pa.field("accuracy", pa.float64(), nullable=False),
    pa.field("roc_auc", pa.float64(), nullable=False),
    pa.field("pr_auc", pa.float64(), nullable=False),
])


@dataclass(frozen=True)
class ParquetMetricsTable:
    """Tabulka metrik v souboru parquet, jeden řádek na řádek matice a diagnózu."""

    path: Path

    def write(self, metrics: Sequence[DiagnosisMetrics]) -> None:
        """Zapíše řádky v daném pořadí, nadřazený adresář založí."""
        columns = {
            "row_id": [item.row_id for item in metrics],
            "diagnosis": [item.diagnosis.value for item in metrics],
            "test_notes": [item.test_notes for item in metrics],
            "test_positives": [item.test_positives for item in metrics],
            "precision": [item.metrics.precision for item in metrics],
            "recall": [item.metrics.recall for item in metrics],
            "f1": [item.metrics.f1 for item in metrics],
            "accuracy": [item.metrics.accuracy for item in metrics],
            "roc_auc": [item.metrics.roc_auc for item in metrics],
            "pr_auc": [item.metrics.pr_auc for item in metrics],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), self.path)
