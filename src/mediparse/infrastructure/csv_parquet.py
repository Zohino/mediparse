"""Převod CSV tabulky MIMIC na parquet po blocích: paměť drží jen jeden blok, i když tabulka má gigabajty."""

from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, override

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape

if TYPE_CHECKING:
    from collections.abc import Buffer
    from pathlib import Path

BLOCK_SIZE: Final = 64 * 1024 * 1024
COMPRESSION: Final = "zstd"


def _header(source: Path) -> tuple[str, ...]:
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as stream:
        return tuple(next(csv.reader(stream), ()))


class _QuoteCounter(io.RawIOBase):
    def __init__(self, inner: io.BufferedIOBase) -> None:
        super().__init__()
        self._inner = inner
        self.quotes = 0

    @override
    def readable(self) -> bool:
        return not self.closed

    @override
    def readinto(self, buffer: Buffer) -> int:
        count = self._inner.readinto(buffer) or 0
        self.quotes += bytes(memoryview(buffer)[:count]).count(b'"')
        return count


@dataclass(frozen=True)
class CsvParquetConversion:
    """Bezeztrátová kopie ``.csv.gz`` do jednoho souboru parquet: všechny sloupce string, prázdná hodnota NULL."""

    source: Path
    target: Path
    block_size: int = BLOCK_SIZE

    def convert(self) -> TableShape:
        """Přečte CSV po blocích a každý blok zapíše jako skupinu řádků.

        Returns:
            Počet zapsaných záznamů a sloupce zapsaného parquetu.

        Raises:
            InvalidInputError: Soubor chybí, nejde rozbalit nebo není platné CSV.
        """
        try:
            columns = _header(self.source)
            return self._write(columns)
        except FileNotFoundError as error:
            msg = f"Soubor {self.source} neexistuje."
            raise InvalidInputError(msg) from error
        except (OSError, EOFError, UnicodeDecodeError, pa.ArrowException) as error:
            msg = f"Soubor {self.source} nejde převést na parquet: {error}"
            raise InvalidInputError(msg) from error

    def _write(self, columns: tuple[str, ...]) -> TableShape:
        with gzip.open(self.source, "rb") as stream:
            counter = _QuoteCounter(stream)
            shape = self._write_blocks(counter, columns)
        if counter.quotes % 2:
            msg = f"Soubor {self.source} má neuzavřené pole v uvozovkách."
            raise InvalidInputError(msg)
        return shape

    def _write_blocks(
        self, stream: io.RawIOBase, columns: tuple[str, ...]
    ) -> TableShape:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=self.block_size),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types=dict.fromkeys(columns, pa.string()),
                strings_can_be_null=True,
            ),
        )
        records = 0
        self.target.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(
            self.target, reader.schema, compression=COMPRESSION
        ) as writer:
            for batch in reader:
                writer.write_batch(batch)
                records += batch.num_rows
        return TableShape(records=records, columns=tuple(reader.schema.names))
