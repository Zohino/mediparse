"""Rezerva překladu nad maskovaným textem a průběžné číslování značek."""

# Skript běží samostatně; kontrakt dělení tvoří jeho soukromé funkce.

from __future__ import annotations

import functools
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.note_tokens_support import marker_tokenizer
from tests.support import REPOSITORY, load_script

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


@pytest.fixture
def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte kopii přípravy s malým oknem bez přístupu k Hubu.

    Returns:
        Skript s oknem 25 tokenů a rezervou 1,0.
    """
    directory = tmp_path / "translation"
    shutil.copytree(
        REPOSITORY / "translation",
        directory,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    path = directory / "prepare.py"
    path.write_bytes(
        path
        .read_bytes()
        .replace(b"WINDOW: Final = 2048", b"WINDOW: Final = 25")
        .replace(b"SPLIT_RATIO: Final = 2.2", b"SPLIT_RATIO: Final = 1.0")
    )
    return load_script(path, monkeypatch)


def test_parts_reserve_room_after_masking(prepare: ModuleType) -> None:
    """Maskování vynutí dvě části s rezervou a číslováním přes hranici bloků."""
    tokenizer = marker_tokenizer()
    text = "a ___ ___\n\nb ___ ___"
    # Zkonstruovaná zpráva: originál má 6 tokenů, maskovaná 2 + 4*3 = 14.
    # Režie chatu je 3: v okně 25 zbude originálu 16 >= 6, masce 8 < 14.
    # Každý maskovaný blok má 7 tokenů a zbude mu 25 - (7 + 3) = 15 >= 7.
    assert prepare._fits(tokenizer, text)  # ruff: ignore[private-member-access]
    assert not prepare._note_fits(tokenizer, text)  # ruff: ignore[private-member-access]

    parts = prepare._parts(tokenizer, text)  # ruff: ignore[private-member-access]

    assert parts == ["a [[1]] [[2]]", "b [[3]] [[4]]"]
    assert all(prepare._fits(tokenizer, part) for part in parts)  # ruff: ignore[private-member-access]


@pytest.mark.parametrize("text", ["a b c", "a b c\n\na b c\n\na b c\n\na b c"])
def test_parts_without_markers_keep_original_split(
    prepare: ModuleType, text: str
) -> None:
    """Bez značek je maskování identita a původní dělení se zachová."""
    tokenizer = marker_tokenizer()
    expected = prepare.split(text, functools.partial(prepare._fits, tokenizer))  # ruff: ignore[private-member-access]

    assert prepare._parts(tokenizer, text) == expected  # ruff: ignore[private-member-access]
