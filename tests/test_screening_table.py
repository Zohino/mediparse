"""Pevné schéma a zachování pořadí agregované tabulky screeningu."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.classical_screening import (
    ClassifierFamily,
    ScreeningResult,
    ScreeningUnit,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import BinaryMetrics, Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.parquet_file import write_table
from mediparse.infrastructure.screening_table import (
    PREDICTIONS_SCHEMA,
    SCHEMA,
    ParquetScreeningCheckpoint,
    ParquetScreeningPredictions,
    ParquetScreeningTable,
)


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


@pytest.fixture
def checkpoint(tmp_path: Path) -> ParquetScreeningCheckpoint:
    """Sestaví checkpoint výhradně nad syntetickými cestami.

    Returns:
        Adaptér se samostatnými výstupy.
    """
    return ParquetScreeningCheckpoint(
        tmp_path / "a" / "checkpoint",
        "synthetic-key",
        tmp_path / "b" / "c" / "results.parquet",
        tmp_path / "b" / "c" / "predictions.parquet",
    )


def _unit_rows(fold: int) -> list[ScreeningResult]:
    return [
        ScreeningResult(
            fold=fold,
            vectorizer="word",
            family=ClassifierFamily.LOGREG,
            diagnosis=diagnosis,
            note_ids=("synthetic-a",),
            labels=(True,),
            classifications=(Classification(predicted=True, score=0.75),),
            threshold=0.5,
            converged=True,
            metrics=BinaryMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        for diagnosis in Diagnosis
    ]


def test_finish_concatenates_parts_in_write_order_and_removes_directory(
    checkpoint: ParquetScreeningCheckpoint,
) -> None:
    """Dvě jednotky zachovají deset buněk i predikcí v pořadí zápisu."""
    checkpoint.write(_unit_rows(1))
    checkpoint.write(_unit_rows(0))
    checkpoint.finish()
    for path in (checkpoint.results, checkpoint.predictions):
        table = pq.read_table(path)
        assert table.num_rows == 10
        assert table["fold"].to_pylist() == [1] * 5 + [0] * 5
    assert not checkpoint.directory.exists()


def test_completed_reads_units_of_written_parts(
    checkpoint: ParquetScreeningCheckpoint,
) -> None:
    """Čtení nevytváří adresář a nová instance obnoví obě jednotky."""
    assert checkpoint.completed() == frozenset()
    assert not checkpoint.directory.exists()
    checkpoint.write(_unit_rows(1))
    checkpoint.write(_unit_rows(0))
    assert replace(checkpoint).completed() == frozenset(
        ScreeningUnit(fold, "word", ClassifierFamily.LOGREG) for fold in (1, 0)
    )


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("key", "jiný config, data, kód nebo knihovny"),
        ("corrupt-key", "jiný config, data, kód nebo knihovny"),
        ("invalid-key-type", "jiný config, data, kód nebo knihovny"),
        ("missing-key", "jiný config, data, kód nebo knihovny"),
        ("gap", "řada dílů je přerušená"),
        ("predictions", "řada dílů je přerušená"),
    ],
)
def test_checkpoint_of_other_run_is_refused_untouched(
    checkpoint: ParquetScreeningCheckpoint,
    damage: str,
    reason: str,
) -> None:
    """Cizí klíč a neúplná řada se odmítnou bez změny bajtů."""
    checkpoint.write(_unit_rows(0))
    checkpoint.write(_unit_rows(1))
    if damage == "key":
        (checkpoint.directory / "key.json").write_text(
            '{"key":"other"}', encoding="utf-8"
        )
    elif damage in {"corrupt-key", "invalid-key-type"}:
        (checkpoint.directory / "key.json").write_bytes(
            b"{" if damage == "corrupt-key" else b'{"key": 1}'
        )
    else:
        name = {
            "missing-key": "key.json",
            "gap": "0000.results.parquet",
            "predictions": "0000.predictions.parquet",
        }[damage]
        (checkpoint.directory / name).unlink()
    before = {p.name: p.read_bytes() for p in checkpoint.directory.iterdir()}
    with pytest.raises(
        InvalidInputError,
        match=re.escape(f"{checkpoint.directory} patří jinému běhu ({reason})"),
    ):
        checkpoint.completed()
    assert {p.name: p.read_bytes() for p in checkpoint.directory.iterdir()} == before


def test_finish_with_foreign_file_refuses_before_outputs(
    checkpoint: ParquetScreeningCheckpoint,
) -> None:
    """Cizí soubor odmítne dokončení před jakýmkoli finálním zápisem."""
    checkpoint.write(_unit_rows(0))
    (checkpoint.directory / "foreign").write_bytes(b"foreign")
    before = {p.name: p.read_bytes() for p in checkpoint.directory.iterdir()}
    with pytest.raises(InvalidInputError, match="cizí soubor"):
        checkpoint.finish()
    assert not checkpoint.results.exists()
    assert not checkpoint.predictions.exists()
    assert {p.name: p.read_bytes() for p in checkpoint.directory.iterdir()} == before


def test_predictions_part_without_results_is_rewritten(
    checkpoint: ParquetScreeningCheckpoint,
) -> None:
    """Predikce bez značky dokončení se přepíší další jednotkou."""
    checkpoint.write(_unit_rows(0))
    (checkpoint.directory / "0000.results.parquet").unlink()
    assert checkpoint.completed() == frozenset()
    checkpoint.write(_unit_rows(1))
    checkpoint.finish()
    for path in (checkpoint.results, checkpoint.predictions):
        assert pq.read_table(path)["fold"].to_pylist() == [1] * 5


def test_interrupted_finish_refuses_resume(
    checkpoint: ParquetScreeningCheckpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chyba druhého odstranění ponechá díly bez klíče a znemožní obnovu."""
    checkpoint.write(_unit_rows(0))
    checkpoint.write(_unit_rows(1))
    parts = {p.name: p.read_bytes() for p in checkpoint.directory.glob("*.parquet")}
    original = Path.unlink
    calls = 0

    def interrupted(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError
        original(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(InvalidInputError, match="OSError"):
        checkpoint.finish()
    assert not (checkpoint.directory / "key.json").exists()
    assert {
        p.name: p.read_bytes() for p in checkpoint.directory.glob("*.parquet")
    } == parts
    with pytest.raises(InvalidInputError):
        checkpoint.completed()


def test_unknown_family_in_part_is_refused(
    checkpoint: ParquetScreeningCheckpoint,
) -> None:
    """Neznámá rodina v dílu je doménová chyba."""
    checkpoint.write(_unit_rows(0))
    path = checkpoint.directory / "0000.results.parquet"
    columns = pq.read_table(path).to_pydict()
    columns["family"] = ["nope"] * 5
    write_table(path, columns, SCHEMA)
    with pytest.raises(InvalidInputError, match="neznámou rodinu"):
        checkpoint.completed()


def test_checkpoint_write_refuses_file_as_parent(tmp_path: Path) -> None:
    """Soubor místo rodičovského adresáře se přeloží na doménovou chybu."""
    parent = tmp_path / "file"
    parent.write_bytes(b"synthetic")
    checkpoint = ParquetScreeningCheckpoint(
        parent / "checkpoint",
        "key",
        tmp_path / "results",
        tmp_path / "predictions",
    )
    with pytest.raises(
        InvalidInputError,
        match="NotADirectoryError",
    ):
        checkpoint.write(_unit_rows(0))
