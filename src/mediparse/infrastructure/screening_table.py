"""Agregované výsledky a predikce screeningu v souborech parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa

from mediparse.infrastructure.metrics_table import METRIC_FIELDS, metric_columns
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
    *METRIC_FIELDS,
])

PREDICTIONS_SCHEMA: Final = pa.schema([
    pa.field("fold", pa.int64(), nullable=False),
    pa.field("vectorizer", pa.string(), nullable=False),
    pa.field("family", pa.string(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("label", pa.bool_(), nullable=False),
    pa.field("predicted", pa.bool_(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
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
            **metric_columns([item.metrics for item in results]),
        }
        write_table(self.path, columns, SCHEMA)


@dataclass(frozen=True)
class ParquetScreeningPredictions:
    """Řádkové predikce v pořadí buněk a testovacích zpráv každé buňky."""

    path: Path

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Rozbalí buňky do řádků po zprávách, nadřazený adresář založí."""
        columns = {
            "fold": [cell.fold for cell in results for _ in cell.note_ids],
            "vectorizer": [cell.vectorizer for cell in results for _ in cell.note_ids],
            "family": [cell.family.value for cell in results for _ in cell.note_ids],
            "diagnosis": [
                cell.diagnosis.value for cell in results for _ in cell.note_ids
            ],
            "note_id": [note_id for cell in results for note_id in cell.note_ids],
            "label": [label for cell in results for label in cell.labels],
            "predicted": [
                item.predicted for cell in results for item in cell.classifications
            ],
            "score": [item.score for cell in results for item in cell.classifications],
        }
        write_table(self.path, columns, PREDICTIONS_SCHEMA)
