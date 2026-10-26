"""Připojení DuckDB pro EDA: jediná cesta ke skutečným datům MIMIC v dokumentech."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import duckdb

from mediparse.domain.data_access import check_data_access
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.synthetic_marker import read_marker

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

DEFAULT_MEMORY: Final = "8GB"
MEMORY_VARIABLE: Final = "MEDIPARSE_DUCKDB_MEMORY"
SPILL_DIRECTORY: Final = "duckdb-tmp"


@dataclass(frozen=True)
class DuckDbSettings:
    """Nastavení spojení: paměťový limit a adresář pro přetok na disk."""

    memory_limit: str
    temp_directory: Path


@dataclass(frozen=True)
class EdaDatabase:
    """Tabulky parquet v adresáři jako pohledy DuckDB, dostupné jen přes stráž.

    ``connect`` spojení nevytvoří, dokud neprojde kontrola adresáře a stráž proti
    čtení skutečných dat, takže bez stráže se k datům nejde dostat.
    """

    parquet: Path
    environ: Mapping[str, str]

    def views(self) -> dict[str, Path]:
        """Tabulky adresáře podle jména souboru bez přípony.

        Returns:
            Jméno pohledu a soubor, seřazené podle jména.

        Raises:
            InvalidInputError: Adresář neexistuje nebo neobsahuje žádný parquet.
        """
        if not self.parquet.is_dir():
            msg = f"Adresář s parquetem {self.parquet} neexistuje."
            raise InvalidInputError(msg)
        files = sorted(self.parquet.glob("*.parquet"))
        if not files:
            msg = f"V adresáři {self.parquet} není žádný parquet."
            raise InvalidInputError(msg)
        return {file.stem: file for file in files}

    def check_access(self) -> dict[str, Path]:
        """Ověří adresář a stráž dřív, než se otevře spojení.

        Returns:
            Pohledy z ``views``.

        Raises:
            InvalidInputError: Adresář je prázdný nebo značka syntetiky je neplatná.
            DataAccessError: Skutečná data v prostředí, které je nesmí číst.
        """
        views = self.views()
        check_data_access(
            self.environ,
            synthetic=read_marker(self.parquet) is not None,
            location=str(self.parquet),
        )
        return views

    def settings(self) -> DuckDbSettings:
        """Paměť z proměnné prostředí (prázdná znamená výchozí) a spill vedle adresáře.

        Returns:
            Nastavení spojení.
        """
        return DuckDbSettings(
            memory_limit=self.environ.get(MEMORY_VARIABLE) or DEFAULT_MEMORY,
            temp_directory=self.parquet.resolve().parent / SPILL_DIRECTORY,
        )

    @contextmanager
    def connect(self) -> Generator[duckdb.DuckDBPyConnection]:
        """Spojení v paměti s pohledem nad každým parquetem; na konci se zavře.

        Yields:
            Spojení s nastavenou pamětí a spillem.

        Raises:
            InvalidInputError: Adresář je prázdný, značka neplatná nebo se tabulka nedá číst.
            DataAccessError: Skutečná data v prostředí, které je nesmí číst.
        """
        views = self.check_access()
        settings = self.settings()
        settings.temp_directory.mkdir(parents=True, exist_ok=True)
        # Absolutní jednotka, protože DuckDB nebere procenta a výchozích 80 % RAM by
        # udusilo Python a Quarto; spill zůstává pod resources/mimic (deny pravidla).
        con = duckdb.connect(
            config={
                "memory_limit": settings.memory_limit,
                "temp_directory": str(settings.temp_directory),
            }
        )
        try:
            for name, path in views.items():
                _create_view(con, name, path)
            yield con
        finally:
            con.close()


def _create_view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    try:
        con.read_parquet(str(path)).create_view(name)
    except duckdb.Error as error:
        msg = f"Tabulka {path} nejde číst: {type(error).__name__}."
        # Text výjimky nese hodnoty buněk z MIMIC; nesmí do hlášky ani do logu, a proto
        # jen její typ a ``from None``.
        raise InvalidInputError(msg) from None
