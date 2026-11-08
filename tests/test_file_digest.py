"""Otisk bajtů souboru a srozumitelné odmítnutí chyb čtení."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.file_digest import file_sha256

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_file_is_refused_with_path(tmp_path: Path) -> None:
    """Chybějící soubor má vlastní diagnostiku s cestou."""
    path = tmp_path / "missing.parquet"

    with pytest.raises(InvalidInputError) as error:
        file_sha256(path)

    assert str(error.value) == f"Soubor {path} neexistuje."


def test_directory_is_refused_as_unreadable_file(tmp_path: Path) -> None:
    """Adresář místo souboru skončí doménovou chybou čtení s cestou."""
    with pytest.raises(InvalidInputError) as error:
        file_sha256(tmp_path)

    assert str(error.value) == f"Soubor {tmp_path} nelze přečíst."


def test_digest_matches_known_sha256(tmp_path: Path) -> None:
    """Bajty abc mají známý SHA-256 podle coreutils sha256sum."""
    path = tmp_path / "output.parquet"
    path.write_bytes(b"abc")

    assert (
        file_sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
