"""Vazba zdrojového CSV a skutečných bajtů převedeného parquetu."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import ParquetSource
from mediparse.infrastructure.parquet_source import (
    source_path,
    verified_parquet_source,
    write_parquet_source,
)

if TYPE_CHECKING:
    from pathlib import Path

CSV_SHA = hashlib.sha256(b"a,b\n1,x\n").hexdigest()
PARQUET_BYTES = b"PAR1 synthetic parquet bytes PAR1"


def test_write_and_verify_source(tmp_path: Path) -> None:
    """Zápis i čtení vrátí předaný původ a nezávislý otisk bajtů parquetu."""
    parquet = tmp_path / "table.parquet"
    parquet.write_bytes(PARQUET_BYTES)
    expected = ParquetSource(
        source="table.csv.gz",
        source_sha256=CSV_SHA,
        sha256=hashlib.sha256(PARQUET_BYTES).hexdigest(),
    )

    written = write_parquet_source(
        parquet, source="table.csv.gz", source_sha256=CSV_SHA
    )

    assert source_path(parquet) == tmp_path / "table.source.json"
    assert written == expected
    assert (
        ParquetSource.model_validate_json(source_path(parquet).read_bytes()) == expected
    )
    assert verified_parquet_source(parquet) == expected


def test_changed_parquet_is_refused_without_content(tmp_path: Path) -> None:
    """Změna jediného bajtu po zápisu vazby je chyba vstupu bez úniku obsahu."""
    parquet = tmp_path / "table.parquet"
    parquet.write_bytes(PARQUET_BYTES)
    write_parquet_source(parquet, source="table.csv.gz", source_sha256=CSV_SHA)
    parquet.write_bytes(b"X" + PARQUET_BYTES[1:])

    with pytest.raises(InvalidInputError, match="převeďte tabulku znovu") as caught:
        verified_parquet_source(parquet)

    assert str(parquet) in str(caught.value)
    assert "synthetic parquet bytes" not in str(caught.value)


@pytest.mark.parametrize("sidecar", [None, "{", '{"source":"tajný obsah"}'])
def test_missing_or_invalid_sidecar_is_refused(
    tmp_path: Path, sidecar: str | None
) -> None:
    """Chybějící nebo neplatná vazba vyžaduje nový převod bez obsahu v hlášce."""
    parquet = tmp_path / "table.parquet"
    parquet.write_bytes(PARQUET_BYTES)
    if sidecar is not None:
        source_path(parquet).write_text(sidecar, encoding="utf-8")

    with pytest.raises(InvalidInputError, match="převeďte tabulku znovu") as caught:
        verified_parquet_source(parquet)

    assert str(parquet) in str(caught.value)
    assert "tajný obsah" not in str(caught.value)
    assert "synthetic parquet bytes" not in str(caught.value)


def test_missing_parquet_is_refused(tmp_path: Path) -> None:
    """Samotná vazba bez parquetu není ověřitelný vstup."""
    parquet = tmp_path / "table.parquet"
    parquet.write_bytes(PARQUET_BYTES)
    write_parquet_source(parquet, source="table.csv.gz", source_sha256=CSV_SHA)
    parquet.unlink()

    with pytest.raises(InvalidInputError, match="převeďte tabulku znovu"):
        verified_parquet_source(parquet)
