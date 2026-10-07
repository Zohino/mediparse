"""Testy dělení zprávy na části a skládání částí zpátky."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from tests.support import REPOSITORY, load_script

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

SCRIPT: Final = REPOSITORY / "translation" / "note_parts.py"
TWO_BLOCKS: Final = 8
SHORT_LINES: Final = 5


def _within(limit: int) -> Callable[[str], bool]:
    return lambda part: len(part) <= limit


@pytest.fixture
def parts(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte skript note_parts.

    Returns:
        Načtený modul.
    """
    return load_script(SCRIPT, monkeypatch)


def test_text_that_fits_is_returned_unchanged(parts: ModuleType) -> None:
    """Text, který se vejde, se vrátí beze změny, aby ID tokenů zůstala stejná."""
    text = "  Hlavička\n\n\nobsah  \n"

    assert parts.split(text, lambda _: True) == [text]


def test_long_text_is_split_on_block_boundaries(parts: ModuleType) -> None:
    """Dlouhý text se dělí na hranicích bloků a každá část splní predikát."""
    text = "a" * 5 + "\n\n" + "b" * 5 + "\n\n" + "c" * 5

    result = parts.split(text, _within(TWO_BLOCKS))

    assert result == ["aaaaa", "bbbbb", "ccccc"]


def test_blocks_are_packed_greedily(parts: ModuleType) -> None:
    """Bloky se balí do jedné části, dokud se vejdou."""
    text = "aaa\n\nbbb\n\nccc"

    result = parts.split(text, _within(TWO_BLOCKS))

    assert result == ["aaa\n\nbbb", "ccc"]


def test_block_that_does_not_fit_is_split_by_lines(parts: ModuleType) -> None:
    """Blok, který se nevejde sám, se dělí po řádcích."""
    text = "head\n\nl1\nl2\nl3\nl4\n\ntail"

    result = parts.split(text, _within(SHORT_LINES))

    assert result == ["head", "l1\nl2", "l3\nl4", "tail"]


def test_line_that_does_not_fit_raises(parts: ModuleType) -> None:
    """Řádek, který se nevejde sám, vyvolá výjimku bez textu zprávy."""
    with pytest.raises(parts.UnsplittableError, match="Řádek") as error:
        parts.split("ok\n\nvery long line", _within(SHORT_LINES))

    assert "very long line" not in str(error.value)


def test_join_restores_text_up_to_normalization(parts: ModuleType) -> None:
    """Spojené části dají původní text s jedním prázdným řádkem mezi bloky."""
    text = "aaa\n\n\n\nbbb\n\nccc"

    result = parts.join(parts.split(text, _within(TWO_BLOCKS)))

    assert result == "aaa\n\nbbb\n\nccc"


def test_part_ids_of_single_part_is_note_id(parts: ModuleType) -> None:
    """Jediná část nese ID zprávy."""
    assert parts.part_ids("n", 1) == ["n"]


def test_part_ids_number_parts_from_one(parts: ModuleType) -> None:
    """Více částí se číslují od jedné za lomítkem."""
    assert parts.part_ids("n", 3) == ["n/1", "n/2", "n/3"]
