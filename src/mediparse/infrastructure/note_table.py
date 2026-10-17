"""Vstupní tabulka kroků pipeline v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote, NoteSubjectMismatchError
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.note import InvalidNoteIdError
from mediparse.infrastructure.parquet_file import read_table, write_table

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
        write_table(self.path, columns, SCHEMA)

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky tabulky v pořadí souboru.

        Returns:
            Zprávy s labely z bool sloupců.

        Raises:
            InvalidInputError: Jazyk je mimo výčet nebo note_id nepatří ``subject_id``.
        """
        table = read_table(self.path, SCHEMA)
        return tuple(self._note(row) for row in table.to_pylist())

    def _note(self, row: dict[str, Any]) -> InputNote:
        language = self._language(row["language"])
        try:
            return InputNote(
                row["note_id"],
                row["subject_id"],
                language,
                row["text"],
                frozenset(diagnosis for diagnosis in Diagnosis if row[diagnosis.value]),
            )
        except (InvalidNoteIdError, NoteSubjectMismatchError) as error:
            msg = (
                f"Soubor {self.path} obsahuje zprávu, jejíž note_id nepatří "
                f"jejímu subject_id: {type(error).__name__}."
            )
            # Text výjimky nese note_id, tedy subject_id pacienta MIMIC; nesmí do hlášky
            # ani do logu, a proto jen její typ a ``from None``.
            raise InvalidInputError(msg) from None

    def _language(self, value: str) -> Language:
        try:
            return Language(value)
        except ValueError as error:
            msg = f"Soubor {self.path} obsahuje neznámý jazyk {value!r}."
            raise InvalidInputError(msg) from error
