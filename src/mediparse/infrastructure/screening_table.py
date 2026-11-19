"""Agregované výsledky a predikce screeningu v souborech parquet s pevným schématem."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pyarrow as pa

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.screening import ClassifierFamily, ScreeningUnit
from mediparse.infrastructure.metrics_table import METRIC_FIELDS, metric_columns
from mediparse.infrastructure.parquet_checkpoint import ParquetParts
from mediparse.infrastructure.parquet_file import read_table, write_arrow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from mediparse.domain.screening import ScreeningResult

logger = logging.getLogger(__name__)

CELL_FIELDS: Final = (
    pa.field("fold", pa.int64(), nullable=False),
    pa.field("vectorizer", pa.string(), nullable=False),
    pa.field("family", pa.string(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
)

SCHEMA: Final = pa.schema([
    *CELL_FIELDS,
    pa.field("test_notes", pa.int64(), nullable=False),
    pa.field("test_positives", pa.int64(), nullable=False),
    pa.field("threshold", pa.float64(), nullable=False),
    pa.field("converged", pa.bool_(), nullable=False),
    *METRIC_FIELDS,
])

PREDICTIONS_SCHEMA: Final = pa.schema([
    *CELL_FIELDS,
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
        write_arrow(self.path, _results_table(results))


@dataclass(frozen=True)
class ParquetScreeningPredictions:
    """Řádkové predikce v pořadí buněk a testovacích zpráv každé buňky."""

    path: Path

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Rozbalí buňky do řádků po zprávách, nadřazený adresář založí."""
        write_arrow(self.path, _predictions_table(results))


def _cell_columns(
    results: Sequence[ScreeningResult], repeat: Callable[[ScreeningResult], int]
) -> dict[str, list[object]]:
    return {
        "fold": [cell.fold for cell in results for _ in range(repeat(cell))],
        "vectorizer": [
            cell.vectorizer for cell in results for _ in range(repeat(cell))
        ],
        "family": [cell.family.value for cell in results for _ in range(repeat(cell))],
        "diagnosis": [
            cell.diagnosis.value for cell in results for _ in range(repeat(cell))
        ],
    }


@dataclass(frozen=True)
class ParquetScreeningCheckpoint:
    """Trvalé díly jednotek a jejich složení do obou výstupů screeningu."""

    directory: Path
    key: str
    results: Path
    predictions: Path
    parts: ParquetParts = field(init=False)

    def __post_init__(self) -> None:
        """Sestaví mechaniku dílů s výsledky jako značkou dokončení."""
        object.__setattr__(
            self,
            "parts",
            ParquetParts(
                self.directory,
                self.key,
                (("predictions", PREDICTIONS_SCHEMA), ("results", SCHEMA)),
                "screeningu",
            ),
        )

    def completed(self) -> frozenset[ScreeningUnit]:
        """Ověří běh a vrátí hotové jednotky.

        Returns:
            Jednotky dokončených dílů výsledků.
        """
        units = set()
        for path in self.parts.completed_parts():
            units.update(_part_units(path))
        if units:
            logger.info("Obnova screeningu: %d hotových jednotek.", len(units))
        return frozenset(units)

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Zapíše predikce a pak výsledky jako značku dokončené jednotky."""
        self.parts.write({
            "predictions": _predictions_table(results),
            "results": _results_table(results),
        })

    def finish(self) -> None:
        """Složí výstupy a odstraní rozpracovaný běh."""
        self.parts.finish({"results": self.results, "predictions": self.predictions})


def _part_units(path: Path) -> frozenset[ScreeningUnit]:
    units = set()
    for row in (
        read_table(path, SCHEMA).select(["fold", "vectorizer", "family"]).to_pylist()
    ):
        try:
            family = ClassifierFamily(row["family"])
        except ValueError:
            msg = f"Soubor {path} obsahuje neznámou rodinu ve sloupci family."
            raise InvalidInputError(msg) from None
        units.add(ScreeningUnit(row["fold"], row["vectorizer"], family))
    return frozenset(units)


def _results_table(results: Sequence[ScreeningResult]) -> pa.Table:
    columns = {
        **_cell_columns(results, lambda _: 1),
        "test_notes": [item.test_notes for item in results],
        "test_positives": [item.test_positives for item in results],
        "threshold": [item.threshold for item in results],
        "converged": [item.converged for item in results],
        **metric_columns([item.metrics for item in results]),
    }
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def _predictions_table(results: Sequence[ScreeningResult]) -> pa.Table:
    columns = {
        **_cell_columns(results, lambda cell: len(cell.note_ids)),
        "note_id": [note_id for cell in results for note_id in cell.note_ids],
        "label": [label for cell in results for label in cell.labels],
        "predicted": [
            item.predicted for cell in results for item in cell.classifications
        ],
        "score": [item.score for cell in results for item in cell.classifications],
    }
    return pa.Table.from_pydict(columns, schema=PREDICTIONS_SCHEMA)
