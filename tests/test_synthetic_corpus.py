"""Adaptér CorpusDirectory: texty po jazycích, odmítnutí neznámého jazyka a stabilní otisk."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.digest import fingerprint
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_notes_of_one_language_are_keyed_by_note_id(tmp_path: Path) -> None:
    """České texty vrací bez přípony a bez anglických zpráv."""
    _write(tmp_path, "en/90000001-DS-1.txt", b"english")
    _write(tmp_path, "cs/90000001-DS-1.txt", "česky".encode())
    _write(tmp_path, "cs/90000002-DS-1.txt", b"dva")

    assert CorpusDirectory(tmp_path).notes(Language.CS) == {
        "90000001-DS-1": "česky",
        "90000002-DS-1": "dva",
    }


def test_language_without_directory_has_no_notes(tmp_path: Path) -> None:
    """Jazyk bez adresáře je prázdný, ne chyba."""
    _write(tmp_path, "en/90000001-DS-1.txt", b"english")

    assert CorpusDirectory(tmp_path).notes(Language.CS) == {}


def test_missing_root_has_no_notes(tmp_path: Path) -> None:
    """Neexistující kořen je prázdný korpus."""
    assert CorpusDirectory(tmp_path / "missing").notes(Language.EN) == {}


def test_unknown_language_directory_is_refused(tmp_path: Path) -> None:
    """Adresář jazyka, který Language nezná, by byl v otisku, ale mimo audit."""
    _write(tmp_path, "en/90000001-DS-1.txt", b"english")
    _write(tmp_path, "de/90000001-DS-1.txt", b"deutsch")
    corpus = CorpusDirectory(tmp_path)

    with pytest.raises(InvalidInputError, match="neznámého jazyka: de"):
        corpus.notes(Language.EN)
    with pytest.raises(InvalidInputError, match="neznámého jazyka: de"):
        corpus.fingerprint()


def test_fingerprint_covers_relative_paths_and_bytes(tmp_path: Path) -> None:
    """Otisk zůstává nad dvojicemi relativní cesta a bajty, aby seděl s commitnutým auditem."""
    _write(tmp_path, "en/90000001-DS-1.txt", b"english")
    _write(tmp_path, "cs/90000001-DS-1.txt", b"cesky")

    assert CorpusDirectory(tmp_path).fingerprint() == fingerprint([
        ("cs/90000001-DS-1.txt", b"cesky"),
        ("en/90000001-DS-1.txt", b"english"),
    ])


def test_note_that_is_not_utf8_is_invalid_input(tmp_path: Path) -> None:
    """Zpráva mimo UTF-8 skončí InvalidInputError s cestou zprávy v korpusu."""
    _write(tmp_path, "en/90000001-DS-1.txt", b"\xff\xfe")

    with pytest.raises(InvalidInputError, match=r"en/90000001-DS-1\.txt"):
        CorpusDirectory(tmp_path).notes(Language.EN)
