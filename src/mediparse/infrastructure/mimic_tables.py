"""Config tabulek MIMIC, které stahuje workflow: oficiální otisky referenčních souborů auditu."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from mediparse.domain.corpus_audit import Sha256

TABLES_PATH: Final = Path("config/mimic_tables.json")
REFERENCE_PROJECT: Final = "mimic-iv-note"


class _Table(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    url: str
    sha256: Sha256

    @property
    def path(self) -> PurePosixPath:
        return PurePosixPath(urlsplit(self.url).path)


class _Tables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mimic_tables: tuple[_Table, ...]


def load_reference_sha256(path: Path) -> dict[str, str]:
    """Přečte otisky tabulek projektu MIMIC-IV-Note, proti kterým běží audit; neplatný obsah vyhodí ValidationError.

    Returns:
        Slovník jméno souboru z URL → SHA-256 ze ``SHA256SUMS.txt`` na PhysioNetu.
    """
    tables = _Tables.model_validate_json(path.read_text(encoding="utf-8"))
    return {
        table.path.name: table.sha256
        for table in tables.mimic_tables
        if REFERENCE_PROJECT in table.path.parts
    }
