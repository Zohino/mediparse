"""Sdílená mechanika atomických dílů parquet a úklidu checkpointu."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.parquet_file import concat_files, write_arrow

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    import pyarrow as pa

KEY_FILE: Final = "key.json"


class CheckpointKey(BaseModel):
    """Otisk vstupů a prostředí rozpracovaného běhu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    key: str


@dataclass(frozen=True)
class ParquetParts:
    """Díly více tabulek; poslední druh označuje dokončený zápis."""

    directory: Path
    key: str
    kinds: tuple[tuple[str, pa.Schema], ...]
    label: str

    def completed_parts(self) -> list[Path]:
        """Ověří klíč a souvislou řadu dokončených dílů.

        Returns:
            Cesty prvního druhu v pořadí zápisu.

        Raises:
            InvalidInputError: Cizí klíč nebo neúplná řada dílů.
        """
        with self._errors():
            if not self.directory.exists():
                return []
            key_path = self.directory / KEY_FILE
            try:
                valid = key_path.is_file() and (
                    CheckpointKey.model_validate_json(key_path.read_bytes()).key
                    == self.key
                )
            except ValueError:
                valid = False
            if not valid:
                msg = "jiný config, data, kód nebo knihovny"
                raise self._invalid(msg)
            marker = self.kinds[-1][0]
            parts = sorted(self.directory.glob(f"*.{marker}.parquet"))
            completed = []
            for index, path in enumerate(parts):
                if path.name != f"{index:04d}.{marker}.parquet" or any(
                    not (self.directory / f"{index:04d}.{kind}.parquet").is_file()
                    for kind, _ in self.kinds
                ):
                    msg = "řada dílů je přerušená"
                    raise self._invalid(msg)
                completed.append(
                    self.directory / f"{index:04d}.{self.kinds[0][0]}.parquet"
                )
            return completed

    def write(self, tables: Mapping[str, pa.Table]) -> None:
        """Zapíše díl každého druhu přes dočasný soubor a značku až nakonec."""
        with self._errors():
            if not self.directory.exists():
                self.directory.mkdir(parents=True)
                write_model(self.directory / KEY_FILE, CheckpointKey(key=self.key))
            index = len(list(self.directory.glob(f"*.{self.kinds[-1][0]}.parquet")))
            for kind, _ in self.kinds:
                target = self.directory / f"{index:04d}.{kind}.parquet"
                temporary = target.with_suffix(".parquet.tmp")
                write_arrow(temporary, tables[kind])
                temporary.replace(target)

    def finish(self, outputs: Mapping[str, Path]) -> None:
        """Složí díly v pořadí a odstraní jen vlastní soubory.

        Raises:
            InvalidInputError: Adresář obsahuje cizí soubor nebo zápis selhal.
        """
        with self._errors():
            files = list(self.directory.iterdir())
            kinds = "|".join(re.escape(kind) for kind, _ in self.kinds)
            if any(
                not re.fullmatch(
                    rf"key\.json|\d{{4}}\.({kinds})\.parquet|.*\.tmp", path.name
                )
                for path in files
            ):
                msg = "adresář obsahuje cizí soubor"
                raise self._invalid(msg)
            for kind, schema in self.kinds:
                concat_files(
                    outputs[kind],
                    sorted(self.directory.glob(f"*.{kind}.parquet")),
                    schema,
                )
            # Bez klíče přerušený úklid odmítne obnovu neúplných dílů.
            (self.directory / KEY_FILE).unlink()
            for path in files:
                if path.name != KEY_FILE:
                    path.unlink()
            self.directory.rmdir()

    def _invalid(self, reason: str) -> InvalidInputError:
        return InvalidInputError(
            f"Adresář rozpracovaného {self.label} {self.directory} patří jinému běhu "
            f"({reason}); smaž ho, nebo zadej jiný."
        )

    @contextmanager
    def _errors(self) -> Generator[None]:
        try:
            yield
        except OSError as error:
            msg = f"Rozpracovaný {self.label} {self.directory} nelze zapsat: {type(error).__name__}."
            raise InvalidInputError(msg) from None
