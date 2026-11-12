"""Vstupní tabulka kroků pipeline v jednom souboru parquet s pevným schématem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.note import NoteIdentityError
from mediparse.infrastructure.parquet_file import (
    note_identity_error,
    read_table,
    write_table,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from mediparse.domain.identity import SubjectId

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
    batch_size: int = 1024

    def write(self, notes: Sequence[InputNote]) -> None:
        """Zapíše fixtury ve schématu tabulky, nadřazený adresář založí.

        V pipeline tabulku zapisuje kohorta přes adaptér ``duckdb_cohort``.
        """
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

    def subjects(self) -> frozenset[SubjectId]:
        """Přečte pouze sloupec pacientů a ověří schéma souboru.

        Returns:
            Všichni různí pacienti tabulky.

        Raises:
            InvalidInputError: Soubor chybí, není parquet nebo má jiné schéma.
        """
        return frozenset(
            subject
            for batch in self._batches(["subject_id"])
            for subject in batch.column("subject_id").to_pylist()
        )

    def read_patients(self, subjects: frozenset[SubjectId]) -> tuple[InputNote, ...]:
        """Proudově přečte a ověří zprávy vybraných pacientů v pořadí souboru.

        Returns:
            Vybrané zprávy ve stejném tvaru jako read().

        Raises:
            InvalidInputError: Soubor nebo vybraná zpráva porušuje vstupní kontrakt.
        """
        selected = pa.array(sorted(subjects), type=pa.int64())
        # Celé čtení drží paměť uvolněných řetězců; parquet filters načte text row groupu.
        return tuple(
            self._note(row)
            for batch in self._batches()
            for row in batch.filter(
                # PyArrow generuje výpočetní funkce za běhu, statický analyzátor je nevidí.
                pc.is_in(batch.column("subject_id"), value_set=selected)  # ty: ignore[unresolved-attribute]
            ).to_pylist()
        )

    def _batches(self, columns: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        try:
            with pq.ParquetFile(self.path) as source:
                if not source.schema_arrow.equals(SCHEMA):
                    msg = f"Soubor {self.path} neodpovídá schématu: čekané sloupce {', '.join(SCHEMA.names)}."
                    raise InvalidInputError(msg)
                yield from source.iter_batches(
                    batch_size=self.batch_size, columns=columns
                )
        except FileNotFoundError as error:
            msg = f"Soubor {self.path} neexistuje."
            raise InvalidInputError(msg) from error
        except pa.ArrowInvalid as error:
            msg = f"Soubor {self.path} neodpovídá schématu: {error}"
            raise InvalidInputError(msg) from error

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
        except NoteIdentityError as error:
            raise note_identity_error(self.path, error) from None

    def _language(self, value: str) -> Language:
        try:
            return Language(value)
        except ValueError:
            msg = f"Soubor {self.path} obsahuje neznámý jazyk ve sloupci language."
            raise InvalidInputError(msg) from None
