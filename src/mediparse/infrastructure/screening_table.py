"""Agregované výsledky a predikce screeningu v souborech parquet s pevným schématem."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from mediparse.domain.classical_screening import ClassifierFamily, ScreeningUnit
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.metrics_table import METRIC_FIELDS, metric_columns
from mediparse.infrastructure.parquet_file import concat_files, read_table, write_table

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from pathlib import Path

    from mediparse.domain.classical_screening import ScreeningResult

logger = logging.getLogger(__name__)
KEY_FILE: Final = "key.json"

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
        columns = {
            **_cell_columns(results, lambda _: 1),
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
            **_cell_columns(results, lambda cell: len(cell.note_ids)),
            "note_id": [note_id for cell in results for note_id in cell.note_ids],
            "label": [label for cell in results for label in cell.labels],
            "predicted": [
                item.predicted for cell in results for item in cell.classifications
            ],
            "score": [item.score for cell in results for item in cell.classifications],
        }
        write_table(self.path, columns, PREDICTIONS_SCHEMA)


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


class CheckpointKey(BaseModel):
    """Otisk vstupů a prostředí rozpracovaného screeningu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    key: str


@dataclass(frozen=True)
class ParquetScreeningCheckpoint:
    """Trvalé díly jednotek a jejich složení do obou výstupů screeningu."""

    directory: Path
    key: str
    results: Path
    predictions: Path

    def completed(self) -> frozenset[ScreeningUnit]:
        """Ověří klíč a souvislost dílů a vrátí hotové jednotky.

        Returns:
            Jednotky uložené v dokončených dílech.

        Raises:
            InvalidInputError: Běh je cizí, poškozený nebo nečitelný.
        """
        with self._errors():
            if not self.directory.exists():
                return frozenset()
            key_path = self.directory / KEY_FILE
            try:
                valid = key_path.is_file() and (
                    CheckpointKey.model_validate_json(key_path.read_bytes()).key
                    == self.key
                )
            except ValueError:
                valid = False
            if not valid:
                reason = "jiný config, data, kód nebo knihovny"
                raise self._invalid(reason)
            parts = sorted(self.directory.glob("*.results.parquet"))
            units = set()
            for index, path in enumerate(parts):
                if (
                    path.name != f"{index:04d}.results.parquet"
                    or not (
                        self.directory / f"{index:04d}.predictions.parquet"
                    ).is_file()
                ):
                    reason = "řada dílů je přerušená"
                    raise self._invalid(reason)
                units.update(_part_units(path))
            if units:
                logger.info("Obnova screeningu: %d hotových jednotek.", len(units))
            return frozenset(units)

    def write(self, results: Sequence[ScreeningResult]) -> None:
        """Trvale zapíše predikce a pak výsledky jako značku dokončené jednotky."""
        with self._errors():
            if not self.directory.exists():
                self.directory.mkdir(parents=True)
                write_model(self.directory / KEY_FILE, CheckpointKey(key=self.key))
            index = len(list(self.directory.glob("*.results.parquet")))
            # Výsledky musí přijít až po predikcích jako značka hotové jednotky.
            for kind, writer in (
                ("predictions", ParquetScreeningPredictions),
                ("results", ParquetScreeningTable),
            ):
                target = self.directory / f"{index:04d}.{kind}.parquet"
                temporary = target.with_suffix(".parquet.tmp")
                writer(temporary).write(results)
                temporary.replace(target)

    def finish(self) -> None:
        """Ověří vlastní soubory, složí výstupy a odstraní rozpracovaný běh.

        Raises:
            InvalidInputError: Adresář obsahuje cizí soubory nebo zápis selhal.
        """
        with self._errors():
            files = list(self.directory.iterdir())
            if any(
                not re.fullmatch(
                    r"key\.json|\d{4}\.(results|predictions)\.parquet|.*\.tmp",
                    path.name,
                )
                for path in files
            ):
                reason = "adresář obsahuje cizí soubor"
                raise self._invalid(reason)
            concat_files(
                self.results, sorted(self.directory.glob("*.results.parquet")), SCHEMA
            )
            concat_files(
                self.predictions,
                sorted(self.directory.glob("*.predictions.parquet")),
                PREDICTIONS_SCHEMA,
            )
            # Bez klíče přerušený úklid odmítne obnovu neúplných dílů.
            (self.directory / KEY_FILE).unlink()
            for path in files:
                if path.name != KEY_FILE:
                    path.unlink()
            self.directory.rmdir()

    def _invalid(self, reason: str) -> InvalidInputError:
        return InvalidInputError(
            f"Adresář rozpracovaného screeningu {self.directory} patří jinému běhu "
            f"({reason}); smaž ho, nebo zadej jiný."
        )

    @contextmanager
    def _errors(self) -> Generator[None]:
        try:
            yield
        except OSError as error:
            msg = f"Rozpracovaný screening {self.directory} nelze zapsat: {type(error).__name__}."
            raise InvalidInputError(msg) from None


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
