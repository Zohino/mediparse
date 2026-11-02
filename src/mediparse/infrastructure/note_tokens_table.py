"""Parquet artefakty délek tokenů s pevným kontraktem Python → R."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.eda_cache import cache_key
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.hf_tokenizers import library_versions
from mediparse.infrastructure.parquet_file import COMPRESSION

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.note_tokens import (
        NoteTokens,
        ParallelTokens,
        SplitSample,
        TokenizerInfo,
        TranslationPrompt,
    )

NOTE_TOKENS_SCHEMA: Final = pa.schema([
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("xlmr", pa.int32(), nullable=False),
    pa.field("mdeberta", pa.int32(), nullable=False),
    pa.field("translategemma", pa.int32(), nullable=False),
])
TOKENIZERS_SCHEMA: Final = pa.schema([
    pa.field("tokenizer", pa.string(), nullable=False),
    pa.field("model", pa.string(), nullable=False),
    pa.field("revision", pa.string(), nullable=False),
    pa.field("window", pa.int32(), nullable=False),
    pa.field("role", pa.string(), nullable=False),
    pa.field("special_tokens", pa.int32(), nullable=False),
    pa.field("transformers", pa.string(), nullable=False),
])
PROMPT_SCHEMA: Final = pa.schema([
    pa.field("prompt_overhead", pa.int32(), nullable=False),
    pa.field("split_ratio", pa.float64(), nullable=False),
    pa.field("sample_limit", pa.int32(), nullable=False),
])
SAMPLE_SCHEMA: Final = pa.schema([
    pa.field("note_table", pa.string(), nullable=False),
    pa.field("note_id", pa.string(), nullable=False),
    pa.field("fits", pa.bool_(), nullable=False),
    pa.field("parts", pa.int32(), nullable=False),
    pa.field("unsplittable", pa.bool_(), nullable=False),
])
PARALLEL_SCHEMA: Final = pa.schema([
    pa.field("pair_id", pa.string(), nullable=False),
    pa.field("tokenizer", pa.string(), nullable=False),
    pa.field("english", pa.int32(), nullable=False),
    pa.field("czech", pa.int32(), nullable=False),
])
ARTIFACTS: Final = (
    ("note_tokens_discharge.parquet", NOTE_TOKENS_SCHEMA),
    ("note_tokens_radiology.parquet", NOTE_TOKENS_SCHEMA),
    ("tokenizers.parquet", TOKENIZERS_SCHEMA),
    ("translation_prompt.parquet", PROMPT_SCHEMA),
    ("translation_sample.parquet", SAMPLE_SCHEMA),
    ("parallel_tokens.parquet", PARALLEL_SCHEMA),
)


@dataclass(frozen=True)
class _TokenWriter:
    writers: Mapping[str, pq.ParquetWriter]

    def notes(self, table: str, rows: Sequence[NoteTokens]) -> None:
        """Zapíše délky zpráv jedné tabulky."""
        self._write(f"note_tokens_{table}.parquet", rows)

    def sample(self, rows: Sequence[SplitSample]) -> None:
        """Zapíše přesné dělení vzorku."""
        self._write("translation_sample.parquet", rows)

    def parallel(self, rows: Sequence[ParallelTokens]) -> None:
        """Zapíše délky paralelních syntetických textů."""
        self._write("parallel_tokens.parquet", rows)

    def settings(
        self, infos: Sequence[TokenizerInfo], prompt: TranslationPrompt
    ) -> None:
        """Zapíše parametry tokenizérů a překladu."""
        self._write("tokenizers.parquet", infos)
        self._write("translation_prompt.parquet", [prompt])

    def _write(
        self,
        name: str,
        rows: Sequence[
            NoteTokens
            | SplitSample
            | ParallelTokens
            | TokenizerInfo
            | TranslationPrompt
        ],
    ) -> None:
        writer = self.writers[name]
        writer.write_table(
            pa.Table.from_pylist([asdict(row) for row in rows], schema=writer.schema)
        )


@dataclass(frozen=True)
class ParquetNoteTokens:
    """Vlastník zapisovačů všech artefaktů skupiny tokenů."""

    directory: Path

    @contextmanager
    def writer(self) -> Generator[_TokenWriter]:
        """Otevře i prázdné soubory s pevným schématem a po výpočtu je zavře.

        Yields:
            Sink tokenových faktů.

        Raises:
            InvalidInputError: Artefakty nelze zapsat.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with ExitStack() as stack:
                writers = {
                    name: stack.enter_context(
                        pq.ParquetWriter(
                            self.directory / name, schema, compression=COMPRESSION
                        )
                    )
                    for name, schema in ARTIFACTS
                }
                yield _TokenWriter(writers)
        except (OSError, pa.ArrowException) as error:
            msg = f"Délky tokenů nelze zapsat: {type(error).__name__}."
            raise InvalidInputError(msg) from None


def tokens_key(
    views: Mapping[str, Path],
    translation: Path,
    english: Path,
    czech: Path,
    infos: Sequence[TokenizerInfo],
) -> str:
    """Otiskne tabulky, skripty, páry, revize a verze tokenizace.

    Returns:
        Klíč skupiny tokens se stejnou konstrukcí jako cache sešitu 9.

    Raises:
        InvalidInputError: Vstup klíče nelze přečíst.
    """
    try:
        inputs = [
            (name, file_sha256(views[name]).encode())
            for name in ("discharge", "radiology")
        ]
        inputs.extend(
            (f"translation/{path.name}", path.read_bytes())
            for path in sorted(translation.glob("*.py"))
        )
        for label, directory in (("english", english), ("czech", czech)):
            inputs.extend(
                (f"{label}/{path.name}", path.read_bytes())
                for path in sorted(directory.glob("*.txt"))
            )
        inputs.append((
            "tokenizers",
            json.dumps([asdict(info) for info in infos], sort_keys=True).encode(),
        ))
        inputs.extend(library_versions())
        return cache_key(inputs)
    except OSError as error:
        msg = f"Vstup klíče cache nelze přečíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None
