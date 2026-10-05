"""Vstupní tabulka kroků pipeline v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.smoketest_input import InputNote

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky tabulky v pořadí souboru.

        Returns:
            Zprávy s labely z bool sloupců.

        Raises:
            InvalidInputError: Soubor neexistuje, není parquet nebo nemá pevné schéma.
        """
        try:
            table = pq.read_table(self.path)
        except FileNotFoundError as error:
            msg = f"Soubor {self.path} neexistuje."
            raise InvalidInputError(msg) from error
        except pa.ArrowInvalid as error:
            msg = f"Soubor {self.path} neodpovídá schématu: {error}"
            raise InvalidInputError(msg) from error
        if not table.schema.equals(SCHEMA):
            msg = f"Soubor {self.path} neodpovídá schématu tabulky zpráv."
            raise InvalidInputError(msg)
        return tuple(
            InputNote(
                row["note_id"],
                row["subject_id"],
                row["language"],
                row["text"],
                frozenset(diagnosis for diagnosis in Diagnosis if row[diagnosis.value]),
            )
            for row in table.to_pylist()
        )
