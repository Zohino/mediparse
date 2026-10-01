"""Config tabulek MIMIC: oficiální otisky stažených souborů podle jména souboru."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mediparse.infrastructure.mimic_tables import load_pinned_sha256

REPO_TABLES = Path(__file__).parents[1] / "config" / "mimic_tables.json"
NOTE_URL = "https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"


def _tables(path: Path, sha256: str) -> Path:
    path.write_text(
        json.dumps({"mimic_tables": [{"url": NOTE_URL, "sha256": sha256}]}),
        encoding="utf-8",
    )
    return path


def test_checksums_are_keyed_by_file_name_from_url(tmp_path: Path) -> None:
    """Klíčem je jméno souboru z URL, stejně jako ve Snakefile a v záznamu auditu."""
    tables = _tables(tmp_path / "mimic_tables.json", "b" * 64)

    assert load_pinned_sha256(tables) == {"discharge.csv.gz": "b" * 64}


def test_invalid_checksum_is_rejected(tmp_path: Path) -> None:
    """Otisk, který není SHA-256 v hexadecimálním tvaru, config neprojde."""
    tables = _tables(tmp_path / "mimic_tables.json", "B" * 64)

    with pytest.raises(ValidationError):
        load_pinned_sha256(tables)


def test_repository_config_pins_note_tables() -> None:
    """Config v repu připíná otisky obou souborů MIMIC-IV-Note, proti kterým běží audit."""
    pinned = load_pinned_sha256(REPO_TABLES)

    assert {"discharge.csv.gz", "radiology.csv.gz"} <= pinned.keys()
