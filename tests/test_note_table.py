"""Parquet adaptér vstupní tabulky: pevné schéma a text beze změny."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import InputNote
from mediparse.infrastructure.note_table import SCHEMA, ParquetNoteTable

if TYPE_CHECKING:
    from pathlib import Path

TEXT = "Pacientka přijata.\r\nDiagnóza: ___\n"


def test_writes_rows_in_fixed_schema(tmp_path: Path) -> None:
    """Labely jsou bool sloupce v pořadí výčtu, text projde beze změny."""
    path = tmp_path / "missing" / "notes.parquet"
    notes = [
        InputNote("90000001-DS-1", 90000001, "en", TEXT, frozenset({Diagnosis.CKD})),
        InputNote("90000002-DS-1", 90000002, "en", "b", frozenset()),
    ]

    ParquetNoteTable(path).write(notes)

    table = pq.read_table(path)
    assert table.schema.equals(SCHEMA)
    assert table.column_names == [
        "note_id",
        "subject_id",
        "language",
        "text",
        *(diagnosis.value for diagnosis in Diagnosis),
    ]
    assert table.to_pylist()[0] == {
        "note_id": "90000001-DS-1",
        "subject_id": 90000001,
        "language": "en",
        "text": TEXT,
        "diabetes": False,
        "ckd": True,
        "heart_failure": False,
        "atrial_fibrillation": False,
        "aki": False,
    }
    assert table.column("note_id").to_pylist() == ["90000001-DS-1", "90000002-DS-1"]


def test_rewrites_previous_table(tmp_path: Path) -> None:
    """Druhý zápis nahradí obsah, nepřipojí řádky."""
    path = tmp_path / "notes.parquet"
    note = InputNote("90000001-DS-1", 90000001, "en", "a", frozenset())
    table = ParquetNoteTable(path)

    table.write([note, note])
    table.write([note])

    assert pq.read_table(path).num_rows == 1
