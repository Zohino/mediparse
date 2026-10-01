"""Config tabulek MIMIC, které stahuje workflow: oficiální otisky souborů podle jména."""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from mediparse.domain.corpus_audit import Sha256

TABLES_PATH: Final = Path("config/mimic_tables.json")


class _Table(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    url: str
    sha256: Sha256


class _Tables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mimic_tables: tuple[_Table, ...]


def load_pinned_sha256(path: Path) -> dict[str, str]:
    """Přečte otisky, podle kterých workflow ověřilo stažené tabulky; neplatný obsah vyhodí ValidationError.

    Returns:
        Slovník jméno souboru z URL → SHA-256 ze ``SHA256SUMS.txt`` na PhysioNetu.
    """
    tables = _Tables.model_validate_json(path.read_text(encoding="utf-8"))
    return {
        posixpath.basename(table.url): table.sha256 for table in tables.mimic_tables
    }
