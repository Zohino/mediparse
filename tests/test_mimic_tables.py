"""Config tabulek MIMIC: soubory ke stažení a otisky souborů MIMIC-IV-Note, proti kterým běží audit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.mimic_tables import (
    load_mimic_tables,
    load_reference_sha256,
)
from tests.support import REPOSITORY_TABLES

if TYPE_CHECKING:
    from pathlib import Path

NOTE_URL = "https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
HOSP_URL = "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"


def _tables(path: Path, *tables: tuple[str, str]) -> Path:
    entries = [{"url": url, "sha256": sha256} for url, sha256 in tables]
    path.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")
    return path


def test_reference_is_note_project_keyed_by_file_name(tmp_path: Path) -> None:
    """Reference jsou tabulky MIMIC-IV-Note pod jménem souboru z URL; hosp mezi ně nepatří."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (HOSP_URL, "f" * 64))

    assert load_reference_sha256(tables) == {"discharge.csv.gz": "b" * 64}


def test_invalid_checksum_is_rejected(tmp_path: Path) -> None:
    """Otisk, který není SHA-256 v hexadecimálním tvaru, config neprojde."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "B" * 64))

    with pytest.raises(InvalidInputError, match=r"t\.json"):
        load_reference_sha256(tables)


def test_duplicate_file_name_is_rejected(tmp_path: Path) -> None:
    """Dvě tabulky se stejným jménem souboru by se při stažení i v referenci přepsaly."""
    other = NOTE_URL.replace("2.2", "2.1")
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (other, "c" * 64))

    with pytest.raises(InvalidInputError, match=r"discharge\.csv\.gz"):
        load_reference_sha256(tables)


def test_tables_cover_every_table_with_url_and_checksum(tmp_path: Path) -> None:
    """Tabulky ke stažení jsou všechny tabulky configu, i ty mimo referenci auditu."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (HOSP_URL, "f" * 64))

    assert {
        name: (table.url, table.sha256)
        for name, table in load_mimic_tables(tables).items()
    } == {
        "discharge.csv.gz": (NOTE_URL, "b" * 64),
        "admissions.csv.gz": (HOSP_URL, "f" * 64),
    }


def test_repository_reference_is_discharge_and_radiology() -> None:
    """Config v repu vymezuje referenci auditu přesně na discharge a radiology."""
    reference = load_reference_sha256(REPOSITORY_TABLES)

    assert reference.keys() == {"discharge.csv.gz", "radiology.csv.gz"}
