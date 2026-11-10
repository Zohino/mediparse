"""Pevné schéma a zachování pořadí agregované tabulky screeningu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.classical_screening import ClassifierFamily, ScreeningResult
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics, Classification
from mediparse.infrastructure.screening_table import (
    PREDICTIONS_SCHEMA,
    SCHEMA,
    ParquetScreeningPredictions,
    ParquetScreeningTable,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("empty_field", ["note_ids", "labels", "classifications"])
def test_cell_rejects_mismatched_lengths(empty_field: str) -> None:
    """Každá n-tice musí mít stejnou délku jako zbývající dvě."""
    with pytest.raises(
        ValueError,
        match=r"^Identity, labely a predikce buňky musí mít stejné délky\.$",
    ):
        ScreeningResult(
            fold=0,
            vectorizer="word",
            family=ClassifierFamily.LOGREG,
            diagnosis=Diagnosis.CKD,
            note_ids=() if empty_field == "note_ids" else ("synthetic-a",),
            labels=() if empty_field == "labels" else (True,),
            classifications=(
                ()
                if empty_field == "classifications"
                else (Classification(predicted=True, score=0.75),)
            ),
            threshold=0.5,
            converged=True,
            metrics=BinaryMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )


def test_written_schema_and_rows_match_contract(tmp_path: Path) -> None:
    """Parquet zachová pořadí a hodnoty vstupů i přesné nenulovatelné typy kontraktu."""
    path = tmp_path / "out" / "screening.parquet"
    rows = [
        ScreeningResult(
            fold=fold,
            vectorizer="word",
            family=ClassifierFamily.LOGREG,
            diagnosis=Diagnosis.CKD,
            note_ids=tuple(f"synthetic-{index}" for index in range(40)),
            labels=(True,) * 12 + (False,) * 28,
            classifications=(Classification(predicted=True, score=0.75),) * 40,
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


def test_predictions_preserve_schema_cell_order_and_note_order(tmp_path: Path) -> None:
    """Tři řádky dvou buněk přenesou přesné hodnoty i nenulovatelné typy."""
    path = tmp_path / "rows" / "predictions.parquet"
    rows = [
        ScreeningResult(
            fold=fold,
            vectorizer=vectorizer,
            family=family,
            diagnosis=diagnosis,
            note_ids=note_ids,
            labels=labels,
            classifications=classifications,
            threshold=0.5,
            converged=True,
            metrics=BinaryMetrics(0.5, 0.25, 0.75, 0.875, 0.9, 0.8),
        )
        for fold, vectorizer, family, diagnosis, note_ids, labels, classifications in [
            (
                1,
                "word",
                ClassifierFamily.LOGREG,
                Diagnosis.CKD,
                ("synthetic-b", "synthetic-a"),
                (True, False),
                (
                    Classification(predicted=False, score=0.2),
                    Classification(predicted=True, score=0.9),
                ),
            ),
            (
                0,
                "char",
                ClassifierFamily.LINEAR_SVM,
                Diagnosis.DIABETES,
                ("synthetic-a",),
                (True,),
                (Classification(predicted=True, score=0.7),),
            ),
        ]
    ]
    expected_schema = pa.schema([
        pa.field(name, kind, nullable=False)
        for name, kind in [
            ("fold", pa.int64()),
            ("vectorizer", pa.string()),
            ("family", pa.string()),
            ("diagnosis", pa.string()),
            ("note_id", pa.string()),
            ("label", pa.bool_()),
            ("predicted", pa.bool_()),
            ("score", pa.float64()),
        ]
    ])

    ParquetScreeningPredictions(path).write(rows)

    table = pq.read_table(path)
    assert table.schema == expected_schema == PREDICTIONS_SCHEMA
    assert table.to_pylist() == [
        {
            "fold": 1,
            "vectorizer": "word",
            "family": "logreg",
            "diagnosis": "ckd",
            "note_id": "synthetic-b",
            "label": True,
            "predicted": False,
            "score": 0.2,
        },
        {
            "fold": 1,
            "vectorizer": "word",
            "family": "logreg",
            "diagnosis": "ckd",
            "note_id": "synthetic-a",
            "label": False,
            "predicted": True,
            "score": 0.9,
        },
        {
            "fold": 0,
            "vectorizer": "char",
            "family": "linear_svm",
            "diagnosis": "diabetes",
            "note_id": "synthetic-a",
            "label": True,
            "predicted": True,
            "score": 0.7,
        },
    ]
