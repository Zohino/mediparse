"""Agregované výsledky screeningu v souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa

from mediparse.infrastructure.parquet_file import write_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.classical_screening import ScreeningResult

SCHEMA: Final = pa.schema([
    pa.field("fold", pa.int64(), nullable=False),
    pa.field("vectorizer", pa.string(), nullable=False),
    pa.field("family", pa.string(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
    pa.field("test_notes", pa.int64(), nullable=False),
    pa.field("test_positives", pa.int64(), nullable=False),
    pa.field("threshold", pa.float64(), nullable=False),
    pa.field("converged", pa.bool_(), nullable=False),
    pa.field("precision", pa.float64(), nullable=False),
    pa.field("recall", pa.float64(), nullable=False),
    pa.field("f1", pa.float64(), nullable=False),
    pa.field("accuracy", pa.float64(), nullable=False),
    pa.field("roc_auc", pa.float64(), nullable=False),
    pa.field("pr_auc", pa.float64(), nullable=False),
])


@dataclass(frozen=True)
class ParquetScreeningTable:
    """Tabulka buněk screeningu v pořadí vnějšího foldu a mřížky."""

    path: Path

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Zapíše řádky v daném pořadí, nadřazený adresář založí."""
        columns = {
            "fold": [item.fold for item in results],
            "vectorizer": [item.vectorizer for item in results],
            "family": [item.family.value for item in results],
            "diagnosis": [item.diagnosis.value for item in results],
            "test_notes": [item.test_notes for item in results],
            "test_positives": [item.test_positives for item in results],
            "threshold": [item.threshold for item in results],
            "converged": [item.converged for item in results],
            "precision": [item.metrics.precision for item in results],
            "recall": [item.metrics.recall for item in results],
            "f1": [item.metrics.f1 for item in results],
            "accuracy": [item.metrics.accuracy for item in results],
            "roc_auc": [item.metrics.roc_auc for item in results],
            "pr_auc": [item.metrics.pr_auc for item in results],
        }
        write_table(self.path, columns, SCHEMA)
