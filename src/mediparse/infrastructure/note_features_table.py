"""Parquet kontrakty, zápis měření zpráv a klíč jejich cache."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.eda_cache import cache_key
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.parquet_file import COMPRESSION

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from mediparse.domain.note_features import (
        AcronymCount,
        NotePreprocessing,
        VariantCount,
    )
    from mediparse.domain.note_sections import SectionSpan

SECTIONS_FILE: Final = "note_sections.parquet"
PREPROCESSING_FILE: Final = "note_preprocessing.parquet"
ACRONYMS_FILE: Final = "note_acronyms.parquet"
VARIANTS_FILE: Final = "note_variants.parquet"
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
VARIANTS_SCHEMA: Final = pa.schema([
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("variant", pa.string(), nullable=False),
    pa.field("diagnosis", pa.string(), nullable=False),
    pa.field("occurrences", pa.int32(), nullable=False),
])
ARTIFACTS: Final = (
    (SECTIONS_FILE, SECTIONS_SCHEMA),
    (PREPROCESSING_FILE, PREPROCESSING_SCHEMA),
    (ACRONYMS_FILE, ACRONYMS_SCHEMA),
    (VARIANTS_FILE, VARIANTS_SCHEMA),
)


@dataclass(frozen=True)
class _BatchWriter:
    writers: tuple[pq.ParquetWriter, ...]

    def write(
        self,
        sections: Sequence[SectionSpan],
        preprocessing: Sequence[NotePreprocessing],
        acronyms: Sequence[AcronymCount],
        variants: Sequence[VariantCount],
    ) -> None:
        """Zapíše dávku do otevřených souborů podle deklarovaných schémat."""
        for writer, (_, schema), rows in zip(
            self.writers,
            ARTIFACTS,
            (sections, preprocessing, acronyms, variants),
            strict=True,
        ):
            writer.write_table(
                pa.Table.from_pylist([asdict(row) for row in rows], schema=schema)
            )


@dataclass(frozen=True)
class ParquetNoteFeatures:
    """Vlastník životnosti čtyř zapisovačů v adresáři artefaktů."""

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


def note_features_key(
    discharge: Path, lexicon: Path, candidates: Path, variants: Path
) -> str:
    """Otisk vstupu, konfigurace, formátu, Python kódu a verze scikit-learn.

    Returns:
        SHA-256 klíč cache.

    Raises:
        InvalidInputError: Vstup klíče nelze přečíst.
    """
    try:
        return cache_key([
            ("discharge", file_sha256(discharge).encode()),
            ("lexicon", lexicon.read_bytes()),
            ("candidates", candidates.read_bytes()),
            ("variants", variants.read_bytes()),
            ("scikit-learn", sklearn.__version__.encode()),
        ])
    except OSError as error:
        msg = f"Vstup klíče cache nelze přečíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None
