"""Config tabulek MIMIC: soubory ke stažení a jejich otisky."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.mimic_tables import (
    load_mimic_tables,
    load_source_digests,
)

if TYPE_CHECKING:
    from pathlib import Path

NOTE_URL = "https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
HOSP_URL = "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"


def _tables(path: Path, *tables: tuple[str, str]) -> Path:
    entries = [{"url": url, "sha256": sha256} for url, sha256 in tables]
    path.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")
    return path


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
