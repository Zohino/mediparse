"""Adaptér CorpusDirectory: texty po jazycích, odmítnutí neznámého jazyka a stabilní otisk."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.digest import fingerprint
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.synthetic.infrastructure.synthetic_corpus import (
    CORPUS_ROOT,
    CorpusDirectory,
)

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
    corpus = CorpusDirectory(tmp_path / "missing")

    assert corpus.notes(Language.EN) == {}
    assert corpus.fingerprint() is None


def test_unknown_language_directory_is_refused(tmp_path: Path) -> None:
    """Adresář jazyka, který Language nezná, by byl v otisku, ale mimo audit."""
    _write(tmp_path, "en/90000001-DS-1.txt", b"english")
    _write(tmp_path, "de/90000001-DS-1.txt", b"deutsch")
    corpus = CorpusDirectory(tmp_path)

    with pytest.raises(InvalidInputError, match="nepovolené cesty: de") as raised:
        corpus.notes(Language.EN)
    assert "Adresář neznámého jazyka:" not in str(raised.value)
    with pytest.raises(InvalidInputError, match="nepovolené cesty: de"):
        corpus.fingerprint()


@pytest.mark.parametrize(
    ("relative", "directory"),
    [
        ("en/extra.csv", False),
        ("en/sub/x.txt", False),
        ("notes.parquet", False),
        ("extra", True),
        ("audit.json", True),
        ("en", False),
        ("en/note.txt", True),
    ],
)
def test_unexpected_corpus_path_is_refused(
    tmp_path: Path, relative: str, *, directory: bool
) -> None:
    """Otisk i čtení odmítnou cizí cestu v celém korpusu a uvedou ji v hlášce."""
    if directory:
        (tmp_path / relative).mkdir(parents=True)
    else:
        _write(tmp_path, relative, b"unexpected")
    corpus = CorpusDirectory(tmp_path)

    with pytest.raises(InvalidInputError, match=re.escape(relative)):
        corpus.fingerprint()
    with pytest.raises(InvalidInputError, match=re.escape(relative)):
        corpus.notes(Language.EN)


def test_expected_corpus_layout_is_accepted(tmp_path: Path) -> None:
    """Čtyři soubory metadat a zprávy en/cs odpovídají povolenému tvaru."""
    for name in ("audit.json", "provenance.json", "plans.jsonl", "labels.csv"):
        _write(tmp_path, name, b"metadata")
    _write(tmp_path, "en/note.txt", b"english")
    _write(tmp_path, "cs/note.txt", b"cesky")
    corpus = CorpusDirectory(tmp_path)

    assert corpus.fingerprint() is not None
    assert corpus.notes(Language.EN) == {"note": "english"}
    assert corpus.notes(Language.CS) == {"note": "cesky"}


def test_repository_corpus_layout_is_accepted() -> None:
    """Skutečný syntetický korpus repa splňuje kontrolu celého stromu."""
    assert CorpusDirectory(CORPUS_ROOT).fingerprint() is not None


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


@pytest.mark.parametrize("language", list(Language))
def test_note_path_is_language_directory_and_note_id(
    tmp_path: Path, language: Language
) -> None:
    """Cesta zprávy je ``<kořen>/<jazyk>/<note_id>.txt``; disk se nečte."""
    root = tmp_path / "neexistuje"

    path = CorpusDirectory(root).note_path(language, "9000-DS-1")

    assert path == root / language.value / "9000-DS-1.txt"


@pytest.mark.parametrize("language", list(Language))
def test_text_written_to_note_path_is_read_back_by_notes(
    tmp_path: Path, language: Language
) -> None:
    """Skládání cesty a čtení zpráv stojí na stejném formátu souboru."""
    corpus = CorpusDirectory(tmp_path)
    path = corpus.note_path(language, "9000-DS-1")
    path.parent.mkdir(parents=True)
    path.write_text("text", encoding="utf-8")

    assert corpus.notes(language) == {"9000-DS-1": "text"}
