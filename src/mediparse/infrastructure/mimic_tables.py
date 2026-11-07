"""Config tabulek MIMIC: soubory ke stažení a jejich otisky."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Final, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator

from mediparse.domain.digest import Sha256
from mediparse.domain.mimic_table import table_name
from mediparse.infrastructure.input_file import parse_file

TABLES_PATH: Final = Path("config/mimic_tables.json")


class MimicTable(BaseModel):
    """Zdrojová tabulka MIMIC: URL souboru a očekávaný otisk CSV."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    url: str
    sha256: Sha256

    @property
    def path(self) -> PurePosixPath:
        """Cesta souboru v adrese zdroje."""
        return PurePosixPath(urlsplit(self.url).path)


class MimicTables(BaseModel):
    """Config zdrojových tabulek s jednoznačnými jmény souborů."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mimic_tables: tuple[MimicTable, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        # Jméno tabulky pojmenuje inventář i parquet; prostý a komprimovaný soubor téže
        # tabulky by se v nich přepsaly, i když se jména souborů liší.
        names = Counter(table_name(table.path.name) for table in self.mimic_tables)
        duplicates = sorted(name for name, count in names.items() if count > 1)
        if duplicates:
            msg = f"Config obsahuje duplicitní jména tabulek: {', '.join(duplicates)}."
            raise ValueError(msg)
        return self


def load_mimic_tables(path: Path) -> dict[str, MimicTable]:
    """Přečte tabulky MIMIC, které workflow stahuje.

    Returns:
        Slovník jméno souboru z URL → tabulka, v pořadí configu.
    """
    tables = parse_file(path, MimicTables.model_validate_json)
    return {table.path.name: table for table in tables.mimic_tables}
