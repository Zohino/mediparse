"""Inventář tabulky MIMIC ze validace: počet záznamů a sloupce, které naměřila nezávisle na převodu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from mediparse.domain.mimic_table import TableShape
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


class _Inventory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    bytes: int
    records: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class InventoryFile:
    """JSON inventář jedné tabulky, který zapisuje ``inventory.sh``."""

    path: Path

    def read(self) -> TableShape:
        """Přečte tvar tabulky z inventáře.

        Returns:
            Počet záznamů a sloupce tabulky.
        """
        inventory = parse_file(self.path, _Inventory.model_validate_json)
        return TableShape(records=inventory.records, columns=inventory.columns)
