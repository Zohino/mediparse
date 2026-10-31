"""Připojení DuckDB pro EDA: jediná cesta ke skutečným datům MIMIC v dokumentech."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import duckdb

from mediparse.domain.data_access import check_data_access
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.duckdb_parquet import (
    SPILL_DIRECTORY,
    create_parquet_view,
)
from mediparse.infrastructure.synthetic_marker import MARKER_NAME, read_marker

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

DEFAULT_MEMORY: Final = "8GB"
MEMORY_VARIABLE: Final = "MEDIPARSE_DUCKDB_MEMORY"


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
    # Prostředí může nést tokeny, a proto se nevypisuje v ``repr`` (log, freeze).
    environ: Mapping[str, str] = field(repr=False)

    def guarded_views(self) -> dict[str, Path]:
        """Tabulky adresáře podle jména souboru bez přípony, až po stráži.

        Returns:
            Jméno pohledu a soubor, seřazené podle jména.

        Raises:
            InvalidInputError: Adresář neexistuje, neobsahuje parquet nebo je značka
                syntetiky neplatná.
            DataAccessError: Skutečná data v prostředí, které je nesmí číst.
        """
        # Adresář a parquet se ověřují před stráží, protože jejich hlášky drží testy
        # dokumentů; stráž čte jen značku, ne data, takže pořadí bezpečnost nesnižuje.
        views = self._views()
        check_data_access(
            self.environ,
            synthetic=read_marker(self.parquet) is not None,
            location=str(self.parquet),
            marker=MARKER_NAME,
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
            InvalidInputError: Adresář je prázdný, značka neplatná, paměť neplatná
                nebo se tabulka nedá číst.
            DataAccessError: Skutečná data v prostředí, které je nesmí číst.
        """
        with self._connect_views() as (con, _names):
            yield con

    @contextmanager
    def _connect_views(
        self,
    ) -> Generator[tuple[duckdb.DuckDBPyConnection, tuple[str, ...]]]:
        """Otevře spojení a předá jména z téhož stráženého snímku adresáře.

        Yields:
            Spojení a seřazená jména spravovaných pohledů.
        """
        views = self.guarded_views()
        con = _open(self.settings())
        try:
            for name, path in views.items():
                create_parquet_view(con, name, path)
            yield con, tuple(views)
        finally:
            con.close()

    def _views(self) -> dict[str, Path]:
        if not self.parquet.is_dir():
            msg = f"Adresář s parquetem {self.parquet} neexistuje."
            raise InvalidInputError(msg)
        files = sorted(self.parquet.glob("*.parquet"))
        if not files:
            msg = f"V adresáři {self.parquet} není žádný parquet."
            raise InvalidInputError(msg)
        return {file.stem: file for file in files}


def _open(settings: DuckDbSettings) -> duckdb.DuckDBPyConnection:
    # Absolutní jednotka, protože DuckDB nebere procenta a výchozích 80 % RAM by
    # udusilo Python a Quarto; spill zůstává pod resources/mimic (deny pravidla).
    try:
        return duckdb.connect(
            config={
                "memory_limit": settings.memory_limit,
                "temp_directory": str(settings.temp_directory),
            }
        )
    except duckdb.Error as error:
        msg = (
            f"Nastavení {MEMORY_VARIABLE}={settings.memory_limit!r} "
            f"DuckDB nepřijal: {type(error).__name__}."
        )
        raise InvalidInputError(msg) from None
