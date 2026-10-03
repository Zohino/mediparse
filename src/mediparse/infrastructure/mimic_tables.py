"""Config tabulek MIMIC: soubory, které stahuje workflow, a oficiální otisky referenčních souborů auditu."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Final, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator

from mediparse.domain.corpus_audit import Sha256
from mediparse.infrastructure.input_file import parse_file

TABLES_PATH: Final = Path("config/mimic_tables.json")
REFERENCE_PROJECT: Final = "mimic-iv-note"


class MimicTable(BaseModel):
    """Tabulka MIMIC na PhysioNetu: adresa ke stažení a otisk ze ``SHA256SUMS.txt``."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    url: str
    sha256: Sha256

    @property
    def path(self) -> PurePosixPath:
        """Cesta souboru v adrese PhysioNetu."""
        return PurePosixPath(urlsplit(self.url).path)


class _Tables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mimic_tables: tuple[MimicTable, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = Counter(table.path.name for table in self.mimic_tables)
        duplicates = sorted(name for name, count in names.items() if count > 1)
        if duplicates:
            msg = f"Config obsahuje duplicitní názvy souborů: {', '.join(duplicates)}."
            raise ValueError(msg)
        return self


def load_mimic_tables(path: Path) -> dict[str, MimicTable]:
    """Přečte tabulky MIMIC, které workflow stahuje.

    Returns:
        Slovník jméno souboru z URL → tabulka, v pořadí configu.
    """
    tables = parse_file(path, _Tables.model_validate_json)
    return {table.path.name: table for table in tables.mimic_tables}


def load_reference_sha256(path: Path) -> dict[str, str]:
    """Přečte otisky tabulek projektu MIMIC-IV-Note, proti kterým běží audit.

    Returns:
        Slovník jméno souboru z URL → SHA-256 ze ``SHA256SUMS.txt`` na PhysioNetu.
    """
    return {
        name: table.sha256
        for name, table in load_mimic_tables(path).items()
        if REFERENCE_PROJECT in table.path.parts
    }
