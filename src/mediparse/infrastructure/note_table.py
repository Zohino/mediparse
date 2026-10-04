"""Vstupní tabulka kroků pipeline v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.labels import Diagnosis

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.smoketest_input import InputNote

SCHEMA: Final = pa.schema([
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("subject_id", pa.int64(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    *(pa.field(diagnosis.value, pa.bool_(), nullable=False) for diagnosis in Diagnosis),
])


@dataclass(frozen=True)
class ParquetNoteTable:
    """Tabulka zpráv v souboru parquet; labely jako bool sloupec na diagnózu."""

    path: Path

    def write(self, notes: Sequence[InputNote]) -> None:
        """Zapíše řádky v daném pořadí, nadřazený adresář založí."""
        columns = {
            "note_id": [note.note_id for note in notes],
            "subject_id": [note.subject_id for note in notes],
            "language": [note.language for note in notes],
            "text": [note.text for note in notes],
            **{
                diagnosis.value: [diagnosis in note.labels for note in notes]
                for diagnosis in Diagnosis
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), self.path)
