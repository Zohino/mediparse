"""Inventář tabulky MIMIC ze validace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.infrastructure.mimic_inventory import (
    InventoryFile,
    read_manifest,
    read_manifest_bytes,
)

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


MANIFEST = (
    '{"tables": ['
    '{"file": "discharge.csv.gz", "bytes": 1, "records": 3, "columns": ["b", "a"]},'
    '{"file": "d_icd_diagnoses.csv.gz", "bytes": 2, "records": 4, "columns": ["c"]}'
    "]}"
)


def test_manifest_gives_shapes_by_table_name_with_ordered_columns(
    tmp_path: Path,
) -> None:
    """Manifest dá tvary tabulek pod jménem bez .csv.gz a se sloupci v pořadí souboru."""
    path = tmp_path / "manifest.json"
    path.write_text(MANIFEST)

    assert read_manifest(path) == {
        "discharge": TableShape(records=3, columns=("b", "a")),
        "d_icd_diagnoses": TableShape(records=4, columns=("c",)),
    }


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    """Chybějící manifest je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        read_manifest(tmp_path / "none.json")


def test_invalid_json_manifest_is_refused(tmp_path: Path) -> None:
    """Manifest, který není JSON, je chyba vstupu."""
    path = tmp_path / "manifest.json"
    path.write_text("{ne json")

    with pytest.raises(InvalidInputError, match="neodpovídá"):
        read_manifest(path)


def test_manifest_bytes_by_table_name(tmp_path: Path) -> None:
    """Velikost souboru z manifestu je pod jménem tabulky bez .csv.gz."""
    path = tmp_path / "manifest.json"
    path.write_text(MANIFEST)

    assert read_manifest_bytes(path) == {"discharge": 1, "d_icd_diagnoses": 2}
