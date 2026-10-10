"""Parquet adaptér predikcí: pevné schéma, round-trip a odmítnutí cizího souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.evaluation import Prediction
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.prediction_table import SCHEMA, ParquetPredictionTable

if TYPE_CHECKING:
    from pathlib import Path

PREDICTIONS = (
    Prediction(
        "row", 0, "n2", 7, Diagnosis.DIABETES, truth=True, predicted=True, score=1.25
    ),
    Prediction(
        "row", 0, "n1", 5, Diagnosis.DIABETES, truth=False, predicted=True, score=0.5
    ),
    Prediction(
        "row", 1, "n3", 5, Diagnosis.CKD, truth=False, predicted=False, score=-0.75
    ),
)


def test_round_trip_keeps_order_and_types(tmp_path: Path) -> None:
    """Zápis a čtení vrátí stejné predikce ve stejném pořadí."""
    table = ParquetPredictionTable(tmp_path / "out" / "predictions.parquet")

    table.write(PREDICTIONS)
    read = table.read()

    assert read == PREDICTIONS
    assert isinstance(read[0].diagnosis, Diagnosis)
    assert isinstance(read[0].score, float)
    assert pq.read_table(tmp_path / "out" / "predictions.parquet").schema.equals(SCHEMA)


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        ParquetPredictionTable(tmp_path / "missing.parquet").read()


def test_other_schema_is_invalid_input(tmp_path: Path) -> None:
    """Soubor s jiným schématem je InvalidInputError."""
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"note_id": ["a"]}), path)

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        ParquetPredictionTable(path).read()


def test_unknown_diagnosis_is_invalid_input(tmp_path: Path) -> None:
    """Neznámá diagnóza je InvalidInputError s cestou a hodnotou."""
    path = tmp_path / "predictions.parquet"
    ParquetPredictionTable(path).write(PREDICTIONS)
    table = pq.read_table(path).to_pydict()
    table["diagnosis"][0] = "asthma"
    pq.write_table(pa.Table.from_pydict(table, schema=SCHEMA), path)

    with pytest.raises(InvalidInputError, match=r"predictions\.parquet.*asthma"):
        ParquetPredictionTable(path).read()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_is_invalid_input(tmp_path: Path, score: float) -> None:
    """NaN a nekonečné skóre je InvalidInputError."""
    path = tmp_path / "predictions.parquet"
    ParquetPredictionTable(path).write(PREDICTIONS)
    table = pq.read_table(path).to_pydict()
    table["score"][1] = score
    pq.write_table(pa.Table.from_pydict(table, schema=SCHEMA), path)

    with pytest.raises(InvalidInputError, match="skóre"):
        ParquetPredictionTable(path).read()
