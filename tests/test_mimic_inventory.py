"""Inventář tabulky MIMIC ze validace."""

from __future__ import annotations

import json
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
        '{"file": "t.csv.gz", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 10, "records": 3, "columns": ["a", "b"]}',
        encoding="utf-8",
    )
    assert InventoryFile(path).read() == TableShape(records=3, columns=("a", "b"))
    assert InventoryFile(path).sha256() == "a" * 64


def test_missing_file_is_refused(tmp_path: Path) -> None:
    """Chybějící inventář je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        InventoryFile(tmp_path / "none.json").read()


def test_missing_key_is_refused(tmp_path: Path) -> None:
    """Inventář bez klíče je chyba vstupu."""
    path = tmp_path / "t.json"
    path.write_text(
        '{"file": "t.csv.gz", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 10, "columns": ["a"]}',
        encoding="utf-8",
    )
    with pytest.raises(InvalidInputError, match="records"):
        InventoryFile(path).read()


MANIFEST = (
    '{"tables": ['
    '{"file": "discharge.csv.gz", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 1, "records": 3, "columns": ["b", "a"]},'
    '{"file": "d_icd_diagnoses.csv.gz", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 2, "records": 4, "columns": ["c"]},'
    '{"file": "mimic-iv-bhc.csv", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 5, "records": 6, "columns": ["d"]}'
    "]}"
)


def test_manifest_gives_shapes_by_table_name_with_ordered_columns(
    tmp_path: Path,
) -> None:
    """Manifest dá tvary tabulek pod jménem bez přípony CSV a se sloupci v pořadí souboru."""
    path = tmp_path / "manifest.json"
    path.write_text(MANIFEST, encoding="utf-8")

    assert read_manifest(path) == {
        "discharge": TableShape(records=3, columns=("b", "a")),
        "d_icd_diagnoses": TableShape(records=4, columns=("c",)),
        "mimic-iv-bhc": TableShape(records=6, columns=("d",)),
    }


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    """Chybějící manifest je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        read_manifest(tmp_path / "none.json")


def test_invalid_json_manifest_is_refused(tmp_path: Path) -> None:
    """Manifest, který není JSON, je chyba vstupu."""
    path = tmp_path / "manifest.json"
    path.write_text("{ne json", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá"):
        read_manifest(path)


def test_manifest_bytes_by_table_name(tmp_path: Path) -> None:
    """Velikost souboru z manifestu je pod jménem tabulky bez přípony CSV."""
    path = tmp_path / "manifest.json"
    path.write_text(MANIFEST, encoding="utf-8")

    assert read_manifest_bytes(path) == {
        "discharge": 1,
        "d_icd_diagnoses": 2,
        "mimic-iv-bhc": 5,
    }


@pytest.mark.parametrize("digest", [None, "neplatný otisk", "a" * 63])
def test_missing_or_invalid_digest_is_refused(
    tmp_path: Path, digest: str | None
) -> None:
    """Inventář vyžaduje platný SHA-256 před čtením tvaru i otisku."""
    payload = {"file": "t.csv.gz", "bytes": 10, "records": 3, "columns": ["a"]}
    if digest is not None:
        payload["sha256"] = digest
    path = tmp_path / "t.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidInputError, match="sha256"):
        InventoryFile(path).read()
    with pytest.raises(InvalidInputError, match="sha256"):
        InventoryFile(path).sha256()


def test_legacy_manifest_still_supplies_shapes_and_sizes(tmp_path: Path) -> None:
    """Starý veřejný manifest slouží syntetice do uživatelského přepočtu."""
    payload = json.loads(MANIFEST)
    for table in payload["tables"]:
        del table["sha256"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_manifest(path)["discharge"] == TableShape(records=3, columns=("b", "a"))
    assert read_manifest_bytes(path)["discharge"] == 1
