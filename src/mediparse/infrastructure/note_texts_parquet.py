"""Proudové čtení identifikátorů a textů zpráv z parquetu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


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
