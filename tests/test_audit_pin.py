"""Skládání pinu auditu z configu tabulek MIMIC a configu vzorkovače."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.mimic_tables import load_mimic_tables
from mediparse.synthetic.domain.corpus_audit import AuditPin
from mediparse.synthetic.infrastructure.audit_pin import (
    load_audit_pin,
    load_reference_sha256,
)
from tests.support import REPOSITORY_CONFIG, REPOSITORY_TABLES

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.synthetic.infrastructure.sampler_config import SamplerConfig


NOTE_URL = "https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
HOSP_URL = "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
BHC_URL = (
    "https://physionet.org/files/labelled-notes-hospital-course/1.2.0/mimic-iv-bhc.csv"
)


def _tables(path: Path, *tables: tuple[str, str]) -> Path:
    entries = [{"url": url, "sha256": sha256} for url, sha256 in tables]
    path.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")
    return path


def test_repository_pin_holds_note_reference_and_sampler_labels(
    sampler_config: SamplerConfig,
) -> None:
    """Pin z configů v repu nese otisky jen MIMIC-IV-Note a štítky struktury vzorkovače."""
    pin = load_audit_pin(REPOSITORY_TABLES, REPOSITORY_CONFIG)

    assert pin == AuditPin(
        load_reference_sha256(REPOSITORY_TABLES),
        sampler_config.structure.structure_labels,
    )


@pytest.mark.parametrize("missing", ["tables", "config"])
def test_missing_config_is_invalid_input(tmp_path: Path, missing: str) -> None:
    """Chybějící config tabulek nebo vzorkovače končí InvalidInputError."""
    tables = tmp_path / "none.json" if missing == "tables" else REPOSITORY_TABLES
    config = tmp_path / "none.json" if missing == "config" else REPOSITORY_CONFIG

    with pytest.raises(InvalidInputError):
        load_audit_pin(tables, config)


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


def test_plain_csv_table_is_outside_reference(tmp_path: Path) -> None:
    """Prosté CSV jiného projektu se stahuje, ale referencí auditu není."""
    tables = _tables(tmp_path / "t.json", (NOTE_URL, "b" * 64), (BHC_URL, "a" * 64))

    assert load_mimic_tables(tables)["mimic-iv-bhc.csv"].url == BHC_URL
    assert load_reference_sha256(tables) == {"discharge.csv.gz": "b" * 64}


def test_repository_reference_is_discharge_and_radiology() -> None:
    """Config v repu vymezuje referenci auditu přesně na discharge a radiology."""
    reference = load_reference_sha256(REPOSITORY_TABLES)

    assert reference.keys() == {"discharge.csv.gz", "radiology.csv.gz"}
