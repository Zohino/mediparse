"""Predikce testovacích zpráv v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.evaluation import Prediction
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.parquet_file import read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA: Final = pa.schema([
    pa.field("row_id", pa.string(), nullable=False),
    pa.field("fold", pa.int64(), nullable=False),
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("subject_id", pa.int64(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
    pa.field("truth", pa.bool_(), nullable=False),
    pa.field("predicted", pa.bool_(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])


@dataclass(frozen=True)
class ParquetPredictionTable:
    """Tabulka predikcí v souboru parquet, jeden řádek na zprávu a diagnózu."""

    path: Path

    def write(self, predictions: Sequence[Prediction]) -> None:
        """Zapíše řádky v daném pořadí, nadřazený adresář založí."""
        columns = {
            "row_id": [item.row_id for item in predictions],
            "fold": [item.fold for item in predictions],
            "note_id": [item.note_id for item in predictions],
            "subject_id": [item.subject_id for item in predictions],
            "diagnosis": [item.diagnosis.value for item in predictions],
            "truth": [item.truth for item in predictions],
            "predicted": [item.predicted for item in predictions],
            "score": [item.score for item in predictions],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), self.path)

    def read(self) -> tuple[Prediction, ...]:
        """Přečte řádky tabulky v pořadí souboru.

        Returns:
            Predikce s diagnózou jako výčtem.

        Raises:
            InvalidInputError: Skóre není konečné číslo.
        """
        table = read_table(self.path, SCHEMA)
        if not all(math.isfinite(score) for score in table.column("score").to_pylist()):
            msg = f"Soubor {self.path} má nekonečné nebo nečíselné skóre."
            raise InvalidInputError(msg)
        return tuple(self._prediction(row) for row in table.to_pylist())

    def _prediction(self, row: dict[str, Any]) -> Prediction:
        try:
            diagnosis = Diagnosis(row["diagnosis"])
        except ValueError as error:
            msg = f"Soubor {self.path} obsahuje neznámou diagnózu {row['diagnosis']!r}."
            raise InvalidInputError(msg) from error
        return Prediction(
            row["row_id"],
            row["fold"],
            row["note_id"],
            row["subject_id"],
            diagnosis,
            truth=row["truth"],
            predicted=row["predicted"],
            score=row["score"],
        )
