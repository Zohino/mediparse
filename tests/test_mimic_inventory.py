"""Inventář tabulky MIMIC ze validace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.infrastructure.mimic_inventory import InventoryFile

if TYPE_CHECKING:
    from pathlib import Path


def test_valid_inventory_gives_shape(tmp_path: Path) -> None:
    """Platný inventář vrací počet záznamů a sloupce."""
    path = tmp_path / "t.json"
    path.write_text(
        '{"file": "t.csv.gz", "bytes": 10, "records": 3, "columns": ["a", "b"]}'
    )
    assert InventoryFile(path).read() == TableShape(records=3, columns=("a", "b"))


def test_missing_file_is_refused(tmp_path: Path) -> None:
    """Chybějící inventář je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        InventoryFile(tmp_path / "none.json").read()


def test_missing_key_is_refused(tmp_path: Path) -> None:
    """Inventář bez klíče je chyba vstupu."""
    path = tmp_path / "t.json"
    path.write_text('{"file": "t.csv.gz", "bytes": 10, "columns": ["a"]}')
    with pytest.raises(InvalidInputError, match="records"):
        InventoryFile(path).read()
