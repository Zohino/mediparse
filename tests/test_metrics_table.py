"""Parquet adaptér metrik: pevné schéma a hodnoty z DiagnosisMetrics."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics, DiagnosisMetrics
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.metrics_table import SCHEMA, ParquetMetricsTable

METRICS_FIXTURE_ROWS = (
    DiagnosisMetrics(
        "smoketest",
        Diagnosis.DIABETES,
        40,
        12,
        BinaryMetrics(0.5, 0.25, 0.75, 0.875, 0.9, 0.8),
    ),
    DiagnosisMetrics(
        "smoketest",
        Diagnosis.COPD,
        40,
        10,
        BinaryMetrics(0.6, 0.3, 0.4, 0.85, 0.7, 0.55),
    ),
    DiagnosisMetrics(
        "smoketest",
        Diagnosis.CKD,
        40,
        8,
        BinaryMetrics(0.25, 0.125, 0.5, 0.9, 0.65, 0.45),
    ),
    DiagnosisMetrics(
        "smoketest",
        Diagnosis.PNEUMONIA,
        40,
        20,
        BinaryMetrics(1.0, 0.0, 0.0, 0.5, 0.5, 0.5),
    ),
    DiagnosisMetrics(
        "smoketest",
        Diagnosis.ATRIAL_FIBRILLATION,
        40,
        4,
        BinaryMetrics(0.125, 1.0, 0.2, 0.3, 0.95, 0.9),
    ),
)


def test_shared_metrics_fixture_matches_writer(tmp_path: Path) -> None:
    """Fixture vzniká zápisem METRICS_FIXTURE_ROWS přes ParquetMetricsTable.

    Čte ji i analysis/tests/testthat; při změně schématu ji přegenerujte
    a upravte nezávislou deklaraci konzumenta METRICS_COLUMNS v R.
    """
    path = tmp_path / "metrics.parquet"
    ParquetMetricsTable(path).write(METRICS_FIXTURE_ROWS)

    fixture = Path(__file__).parent / "fixtures" / "contracts" / "metrics.parquet"
    assert pq.read_table(path).equals(pq.read_table(fixture))


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


def test_round_trip_keeps_order_and_types(tmp_path: Path) -> None:
    """Zápis a čtení vrátí stejné metriky ve stejném pořadí s diagnózou jako výčtem."""
    table = ParquetMetricsTable(tmp_path / "out" / "metrics.parquet")
    metrics = (
        DiagnosisMetrics(
            "row",
            Diagnosis.DIABETES,
            40,
            12,
            BinaryMetrics(0.5, 0.25, 0.75, 0.875, 0.9, 0.8),
        ),
        DiagnosisMetrics(
            "row", Diagnosis.CKD, 38, 9, BinaryMetrics(0.6, 0.35, 0.45, 0.5, 0.7, 0.6)
        ),
    )

    table.write(metrics)
    read = table.read()

    assert read == metrics
    assert isinstance(read[0].diagnosis, Diagnosis)


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        ParquetMetricsTable(tmp_path / "missing.parquet").read()


def test_foreign_schema_is_invalid_input(tmp_path: Path) -> None:
    """Parquet s cizím schématem je InvalidInputError."""
    path = tmp_path / "foreign.parquet"
    pq.write_table(pa.table({"x": [1]}), path)

    with pytest.raises(InvalidInputError, match="schématu"):
        ParquetMetricsTable(path).read()


def test_non_parquet_file_is_invalid_input(tmp_path: Path) -> None:
    """Soubor, který není parquet, je InvalidInputError."""
    path = tmp_path / "text.parquet"
    path.write_text("ne parquet", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="schématu"):
        ParquetMetricsTable(path).read()


def test_written_file_is_zstd_compressed(tmp_path: Path) -> None:
    """Tabulka se zapisuje se stejnou kompresí jako ostatní parquet repa."""
    path = tmp_path / "metrics.parquet"
    metrics = DiagnosisMetrics(
        "row", Diagnosis.CKD, 40, 12, BinaryMetrics(0.5, 0.25, 0.75, 0.875, 0.9, 0.8)
    )

    ParquetMetricsTable(path).write([metrics])

    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
