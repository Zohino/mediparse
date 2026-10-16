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

    from tests.conftest import AuditFiles

NAME = "discharge.csv.gz"
LEAKING_VALUE = "10000032x"


def _reference(tmp_path: Path, content: bytes) -> MimicReference:
    path = tmp_path / NAME
    path.write_bytes(gzip.compress(content))
    return MimicReference(path)


def test_reference_reader_keeps_multiline_quoted_text(audit_files: AuditFiles) -> None:
    """Víceřádkový text v uvozovkách se přečte jako jedna zpráva beze změny."""
    text = 'first line\nsecond "quoted" line, with comma'
    path = audit_files.reference(NAME, [("10000032", text)])

    assert list(MimicReference(path).notes()) == [ReferenceNote(10000032, text)]


@pytest.mark.parametrize(
    ("content", "missing"),
    [
        pytest.param(b"subject_id,body\n10000032,hello\n", "text", id="bez-text"),
        pytest.param(b"id,text\n10000032,hello\n", "subject_id", id="bez-subject-id"),
    ],
)
def test_missing_column_is_named(tmp_path: Path, content: bytes, missing: str) -> None:
    """Chybějící sloupec jmenuje hláška; jména sloupců nejsou data."""
    reference = _reference(tmp_path, content)

    with pytest.raises(InvalidInputError, match=missing):
        list(reference.notes())


def test_non_numeric_subject_id_is_not_echoed(tmp_path: Path) -> None:
    """Nečíselné subject_id skončí InvalidInputError a jeho hodnota není v hlášce ani v řetězu."""
    reference = _reference(
        tmp_path, f"subject_id,text\n{LEAKING_VALUE},hello\n".encode()
    )

    with pytest.raises(InvalidInputError, match=NAME) as raised:
        list(reference.notes())

    assert "řádk" in str(raised.value)
    assert LEAKING_VALUE not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"subject_id,text\n10000032\n", id="kratky-radek"),
        pytest.param(b"subject_id,text\n10000032,\xff\xfe\n", id="ne-utf8"),
    ],
)
def test_unreadable_row_is_invalid_input(tmp_path: Path, content: bytes) -> None:
    """Krátký řádek a bajty mimo UTF-8 skončí InvalidInputError se jménem souboru."""
    reference = _reference(tmp_path, content)

    with pytest.raises(InvalidInputError, match=NAME) as raised:
        list(reference.notes())

    assert "řádk" in str(raised.value)


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
