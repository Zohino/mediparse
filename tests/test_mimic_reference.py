"""Adaptér MimicReference: vadný soubor jde jako InvalidInputError a hláška nenese hodnoty buněk."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import ReferenceNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.mimic_reference import MimicReference

if TYPE_CHECKING:
    from pathlib import Path

NAME = "discharge.csv.gz"
LEAKING_VALUE = "10000032x"


def _reference(tmp_path: Path, content: bytes) -> MimicReference:
    path = tmp_path / NAME
    path.write_bytes(gzip.compress(content))
    return MimicReference(path)


def test_valid_file_yields_notes(tmp_path: Path) -> None:
    """Platný soubor dá zprávy beze změny."""
    reference = _reference(tmp_path, b"subject_id,text\n10000032,hello\n")

    assert list(reference.notes()) == [ReferenceNote(10000032, "hello")]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"subject_id,body\n10000032,hello\n", id="bez-sloupce-text"),
        pytest.param(
            f"subject_id,text\n{LEAKING_VALUE},hello\n".encode(), id="nenumericke-id"
        ),
        pytest.param(b"subject_id,text\n10000032,\xff\xfe\n", id="ne-utf8"),
    ],
)
def test_unreadable_content_is_invalid_input(tmp_path: Path, content: bytes) -> None:
    """Chyba obsahu skončí InvalidInputError se jménem souboru a řádkem, ne hodnotou buňky."""
    reference = _reference(tmp_path, content)

    with pytest.raises(InvalidInputError, match=NAME) as raised:
        list(reference.notes())

    assert "řádk" in str(raised.value)
    assert LEAKING_VALUE not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_truncated_gzip_is_invalid_input(tmp_path: Path) -> None:
    """Uříznutý gzip skončí InvalidInputError."""
    body = b"subject_id,text\n" + b"10000032,hello world filler\n" * 5000
    path = tmp_path / NAME
    path.write_bytes(gzip.compress(body)[:-200])

    with pytest.raises(InvalidInputError, match=NAME):
        list(MimicReference(path).notes())


def test_not_gzip_is_invalid_input(tmp_path: Path) -> None:
    """Soubor, který není gzip, skončí InvalidInputError."""
    path = tmp_path / NAME
    path.write_bytes(b"subject_id,text\n10000032,hello\n")

    with pytest.raises(InvalidInputError, match=NAME):
        list(MimicReference(path).notes())
