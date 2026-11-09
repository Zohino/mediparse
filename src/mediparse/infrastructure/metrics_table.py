"""Metriky evaluace v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa

from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics
from mediparse.infrastructure.parquet_file import (
    parse_diagnosis,
    read_table,
    write_table,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

METRIC_FIELDS: Final = (
    pa.field("precision", pa.float64(), nullable=False),
    pa.field("recall", pa.float64(), nullable=False),
    pa.field("f1", pa.float64(), nullable=False),
    pa.field("accuracy", pa.float64(), nullable=False),
    pa.field("roc_auc", pa.float64(), nullable=False),
    pa.field("pr_auc", pa.float64(), nullable=False),
)

SCHEMA: Final = pa.schema([
    pa.field("row_id", pa.string(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
    pa.field("test_notes", pa.int64(), nullable=False),
    pa.field("test_positives", pa.int64(), nullable=False),
    *METRIC_FIELDS,
])


def metric_columns(metrics: Sequence[BinaryMetrics]) -> dict[str, list[float]]:
    """Převede binární metriky na sloupce společné tabulkám evaluace a screeningu.

    Returns:
        Šest metrik ve stejném pořadí jako METRIC_FIELDS, řádky v pořadí vstupu.
    """
    return {
        "precision": [item.precision for item in metrics],
        "recall": [item.recall for item in metrics],
        "f1": [item.f1 for item in metrics],
        "accuracy": [item.accuracy for item in metrics],
        "roc_auc": [item.roc_auc for item in metrics],
        "pr_auc": [item.pr_auc for item in metrics],
    }


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
            **metric_columns([item.metrics for item in metrics]),
        }
        write_table(self.path, columns, SCHEMA)

    def read(self) -> tuple[DiagnosisMetrics, ...]:
        """Přečte řádky tabulky v pořadí souboru.

        Returns:
            Metriky s diagnózou jako výčtem.
        """
        table = read_table(self.path, SCHEMA)
        return tuple(self._metrics(row) for row in table.to_pylist())

    def _metrics(self, row: dict[str, Any]) -> DiagnosisMetrics:
        return DiagnosisMetrics(
            row["row_id"],
            parse_diagnosis(self.path, row["diagnosis"]),
            row["test_notes"],
            row["test_positives"],
            BinaryMetrics(
                row["precision"],
                row["recall"],
                row["f1"],
                row["accuracy"],
                row["roc_auc"],
                row["pr_auc"],
            ),
        )
