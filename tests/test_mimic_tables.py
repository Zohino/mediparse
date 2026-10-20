"""Config tabulek MIMIC: soubory ke stažení a otisky souborů MIMIC-IV-Note, proti kterým běží audit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.mimic_tables import (
    load_mimic_tables,
    load_reference_sha256,
    load_source_digests,
)
from tests.support import REPOSITORY_TABLES

if TYPE_CHECKING:
    from pathlib import Path

NOTE_URL = "https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
HOSP_URL = "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
BHC_URL = (
    "https://physionet.org/files/labelled-notes-hospital-course/1.2.0/mimic-iv-bhc.csv"
)


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

    with pytest.raises(InvalidInputError, match=r"jména tabulek: discharge\."):
        load_reference_sha256(tables)


def test_same_table_from_plain_and_gzip_file_is_rejected(tmp_path: Path) -> None:
    """Prostý a komprimovaný soubor téže tabulky by se přepsaly v parquetu i manifestu."""
    plain = NOTE_URL.removesuffix(".gz")
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (plain, "c" * 64))

    with pytest.raises(InvalidInputError, match=r"jména tabulek: discharge\."):
        load_mimic_tables(tables)


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    """Soubor, který stažení, inventář ani převod neumí přečíst, config neprojde."""
    archive = NOTE_URL.replace(".csv.gz", ".zip")
    tables = _tables(tmp_path / "t.json", (archive, "b" * 64))

    with pytest.raises(InvalidInputError, match=r"discharge\.zip"):
        load_mimic_tables(tables)


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


def test_plain_csv_table_is_outside_reference(tmp_path: Path) -> None:
    """Prosté CSV jiného projektu se stahuje, ale referencí auditu není."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (BHC_URL, "a" * 64))

    assert load_mimic_tables(tables)["mimic-iv-bhc.csv"].url == BHC_URL
    assert load_reference_sha256(tables) == {"discharge.csv.gz": "b" * 64}


def test_repository_reference_is_discharge_and_radiology() -> None:
    """Config v repu vymezuje referenci auditu přesně na discharge a radiology."""
    reference = load_reference_sha256(REPOSITORY_TABLES)

    assert reference.keys() == {"discharge.csv.gz", "radiology.csv.gz"}


def test_source_digests_follow_requested_order(tmp_path: Path) -> None:
    """Otisky jsou ve vstupním pořadí jmen, ne v pořadí configu."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (HOSP_URL, "f" * 64))

    digests = load_source_digests(tables, ["admissions.csv.gz", "discharge.csv.gz"])

    assert list(digests.items()) == [
        ("admissions.csv.gz", "f" * 64),
        ("discharge.csv.gz", "b" * 64),
    ]


def test_missing_source_table_is_rejected_by_name(tmp_path: Path) -> None:
    """Tabulka, kterou config nezná, vyhodí chybný vstup, který ji jmenuje."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64))

    with pytest.raises(InvalidInputError, match=r"admissions\.csv\.gz"):
        load_source_digests(tables, ["discharge.csv.gz", "admissions.csv.gz"])
