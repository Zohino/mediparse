"""Sdílený zápis modelů do JSON a čtení JSON Lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_file import parse_file
from mediparse.infrastructure.json_file import parse_lines, write_model

if TYPE_CHECKING:
    from pathlib import Path


class _Item(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    count: int


def test_write_model_creates_directories_and_indents_by_two(tmp_path: Path) -> None:
    """Chybějící adresáře se založí, odsazení je 2 a soubor končí novým řádkem."""
    path = tmp_path / "build" / "smoketest" / "item.json"

    write_model(path, _Item(name="a", count=1))

    assert path.read_bytes() == b'{\n  "name": "a",\n  "count": 1\n}\n'


def test_write_model_writes_non_ascii_as_utf8_with_lf(tmp_path: Path) -> None:
    """Ne-ASCII text je v UTF-8 (ne escapovaný) a konce řádků jsou LF."""
    path = tmp_path / "item.json"

    write_model(path, _Item(name="žluť", count=1))

    content = path.read_bytes()
    assert "žluť".encode() in content
    assert b"\r" not in content


def test_write_model_overwrites_existing_file(tmp_path: Path) -> None:
    """Existující soubor se přepíše celý."""
    path = tmp_path / "item.json"
    path.write_text("x" * 500, encoding="utf-8")

    write_model(path, _Item(name="a", count=1))

    assert _Item.model_validate_json(path.read_text(encoding="utf-8")).name == "a"


def test_parse_lines_returns_models_in_line_order() -> None:
    """Modely jsou v pořadí řádků."""
    content = '{"name":"b","count":2}\n{"name":"a","count":1}\n'

    assert parse_lines(_Item)(content) == (
        _Item(name="b", count=2),
        _Item(name="a", count=1),
    )


def test_parse_lines_skips_blank_and_whitespace_only_lines() -> None:
    """Prázdný řádek uprostřed, na konci i řádek ze samých mezer se přeskočí."""
    content = '{"name":"a","count":1}\n\n   \n{"name":"b","count":2}\n\n'

    assert [item.name for item in parse_lines(_Item)(content)] == ["a", "b"]


@pytest.mark.parametrize("content", ["", "\n\n  \n"])
def test_parse_lines_of_empty_content_is_empty(content: str) -> None:
    """Bez záznamů vznikne prázdná n-tice."""
    assert parse_lines(_Item)(content) == ()


def test_invalid_line_raises_validation_error() -> None:
    """Neplatný neprázdný řádek se nepřeskočí."""
    with pytest.raises(ValidationError):
        parse_lines(_Item)('{"name":"a"}\n')


def test_invalid_line_becomes_invalid_input_via_parse_file(tmp_path: Path) -> None:
    """``parse_file`` chybu schématu převede na doménovou výjimku."""
    path = tmp_path / "items.jsonl"
    path.write_text('{"name":"a"}\n', encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        parse_file(path, parse_lines(_Item))
