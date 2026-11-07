"""Inventář tabulky MIMIC ze validace: počet záznamů a sloupce, které naměřila nezávisle na převodu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from mediparse.domain.digest import Sha256
from mediparse.domain.mimic_table import TableShape, table_name
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


class _TableSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    file: str
    records: int
    columns: tuple[str, ...]


class _Inventory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    bytes: int
    records: int
    columns: tuple[str, ...]
    sha256: Sha256


class _Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Čtenáři manifestu potřebují jen tvar; inventáře ověřuje striktně `_Inventory`,
    # manifest je jejich spojení, proto projekce další pole (i `sha256`) ignoruje.
    tables: tuple[_TableSummary, ...]


def read_manifest(path: Path) -> dict[str, TableShape]:
    """Přečte veřejný manifest validace: tvar každé tabulky pod jejím jménem.

    Returns:
        Jméno tabulky (`table_name`) a její tvar v pořadí manifestu.
    """
    manifest = parse_file(path, _Manifest.model_validate_json)
    return {
        table_name(table.file): TableShape(records=table.records, columns=table.columns)
        for table in manifest.tables
    }


@dataclass(frozen=True)
class InventoryFile:
    """JSON inventář jedné tabulky, který zapisuje ``inventory.sh``."""

    path: Path

    def sha256(self) -> Sha256:
        """Přečte otisk CSV ověřený při sestavení inventáře.

        Returns:
            SHA-256 zdrojového souboru CSV.
        """
        return parse_file(self.path, _Inventory.model_validate_json).sha256

    def read(self) -> TableShape:
        """Přečte tvar tabulky z inventáře.

        Returns:
            Počet záznamů a sloupce tabulky.
        """
        inventory = parse_file(self.path, _Inventory.model_validate_json)
        return TableShape(records=inventory.records, columns=inventory.columns)
