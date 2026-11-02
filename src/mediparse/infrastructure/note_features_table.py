"""Proudové čtení textů a zápis měření s pevným kontraktem Python → R."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.parquet_file import COMPRESSION

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from mediparse.domain.note_features import AcronymCount, NotePreprocessing
    from mediparse.domain.note_sections import SectionSpan

SECTIONS_FILE: Final = "note_sections.parquet"
PREPROCESSING_FILE: Final = "note_preprocessing.parquet"
ACRONYMS_FILE: Final = "note_acronyms.parquet"
SECTIONS_SCHEMA: Final = pa.schema([
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("position", pa.int32(), nullable=False),
    pa.field("section", pa.string(), nullable=False),
    pa.field("span_chars", pa.int32(), nullable=False),
    pa.field("body_chars", pa.int32(), nullable=False),
    pa.field("body_masked", pa.bool_(), nullable=False),
    pa.field("body_start", pa.int32(), nullable=False),
    pa.field("body_end", pa.int32(), nullable=False),
])
PREPROCESSING_SCHEMA: Final = pa.schema([
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("tokens_raw", pa.int32(), nullable=False),
    pa.field("tokens_preprocessed", pa.int32(), nullable=False),
    pa.field("vocabulary_raw", pa.list_(pa.string()), nullable=False),
    pa.field("vocabulary_preprocessed", pa.list_(pa.string()), nullable=False),
])
ACRONYMS_SCHEMA: Final = pa.schema([
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("acronym", pa.string(), nullable=False),
    pa.field("occurrences", pa.int32(), nullable=False),
])
ARTIFACTS: Final = (
    (SECTIONS_FILE, SECTIONS_SCHEMA),
    (PREPROCESSING_FILE, PREPROCESSING_SCHEMA),
    (ACRONYMS_FILE, ACRONYMS_SCHEMA),
)


@dataclass(frozen=True)
class ParquetNoteTexts:
    """Texty zpráv z parquetu v omezených dávkách."""

    path: Path

    def batches(self) -> Generator[list[tuple[str, str]]]:
        """Čte jen identifikátory a text, bez řazení celé tabulky.

        Yields:
            Dávky v pořadí souboru.

        Raises:
            InvalidInputError: Tabulka nejde číst nebo obsahuje neřetězcové hodnoty.
        """
        try:
            with pq.ParquetFile(self.path) as source:
                for batch in source.iter_batches(
                    batch_size=2000, columns=["note_id", "text"]
                ):
                    yield _text_pairs(batch, self.path.stem)
        except (OSError, pa.ArrowException, KeyError) as error:
            msg = f"Tabulku {self.path.stem} nelze číst: {type(error).__name__}."
            raise InvalidInputError(msg) from None


def _text_pairs(batch: pa.RecordBatch, table: str) -> list[tuple[str, str]]:
    rows = batch.to_pydict()
    pairs = list(zip(rows["note_id"], rows["text"], strict=True))
    if any(not isinstance(value, str) for pair in pairs for value in pair):
        msg = f"Tabulka {table} musí obsahovat note_id a text bez NULL jako řetězce."
        raise InvalidInputError(msg)
    return pairs


@dataclass(frozen=True)
class _BatchWriter:
    writers: tuple[pq.ParquetWriter, ...]

    def write(
        self,
        sections: Sequence[SectionSpan],
        preprocessing: Sequence[NotePreprocessing],
        acronyms: Sequence[AcronymCount],
    ) -> None:
        """Zapíše dávku do otevřených souborů podle deklarovaných schémat."""
        for writer, (_, schema), rows in zip(
            self.writers,
            ARTIFACTS,
            (sections, preprocessing, acronyms),
            strict=True,
        ):
            writer.write_table(
                pa.Table.from_pylist([asdict(row) for row in rows], schema=schema)
            )


@dataclass(frozen=True)
class ParquetNoteFeatures:
    """Vlastník životnosti tří zapisovačů v adresáři artefaktů."""

    directory: Path

    @contextmanager
    def writer(self) -> Generator[_BatchWriter]:
        """Otevře typované soubory i pro prázdný zdroj a všechny je uzavře.

        Yields:
            Sink dávkového výpočtu.

        Raises:
            InvalidInputError: Artefakty nejde zapsat.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with ExitStack() as stack:
                writers = tuple(
                    stack.enter_context(
                        pq.ParquetWriter(
                            self.directory / name, schema, compression=COMPRESSION
                        )
                    )
                    for name, schema in ARTIFACTS
                )
                yield _BatchWriter(writers)
        except (OSError, pa.ArrowException) as error:
            msg = f"Měření zpráv nelze zapsat: {type(error).__name__}."
            raise InvalidInputError(msg) from None
