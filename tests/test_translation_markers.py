"""Testy maskování značek ___ číslovanými [[n]] a jejich obnovy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from tests.support import REPOSITORY, load_script

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT: Final = REPOSITORY / "translation" / "markers.py"


@pytest.fixture
def markers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte skript markers.

    Returns:
        Načtený modul.
    """
    return load_script(SCRIPT, monkeypatch)


def test_mask_numbers_across_parts(markers: ModuleType) -> None:
    """Čísla značek běží napříč částmi zprávy."""
    assert markers.mask(["a ___ b ___", "c ___"]) == ["a [[1]] b [[2]]", "c [[3]]"]


def test_mask_keeps_longer_and_shorter_runs(markers: ModuleType) -> None:
    """Řady čtyř a dvou podtržítek se nemaskují."""
    assert markers.mask(["a ____ b __ c ___"]) == ["a ____ b __ c [[1]]"]


def test_mask_leaves_part_without_marker(markers: ModuleType) -> None:
    """Část bez značky zůstane beze změny."""
    assert markers.mask(["plain", "x ___"]) == ["plain", "x [[1]]"]


def test_restore_tolerates_spaces_and_extra_numbers(markers: ModuleType) -> None:
    """Obnova zvládne mezery uvnitř značky i číslo navíc."""
    assert markers.restore("a [[ 3 ]] b [[1]] c [[ 17]]") == "a ___ b ___ c ___"


def test_problems_none_for_swapped_order(markers: ModuleType) -> None:
    """Přehozené pořadí čísel není problém."""
    problems = markers.placeholder_problems("[[2]] a [[1]]", 2)
    assert problems == markers.PlaceholderProblems((), (), ())


def test_problems_missing(markers: ModuleType) -> None:
    """Chybějící číslo se ohlásí."""
    problems = markers.placeholder_problems("[[1]] [[3]]", 3)
    assert problems.missing == (2,)
    assert problems.extra == ()
    assert problems.duplicated == ()


def test_problems_extra_includes_zero_and_beyond(markers: ModuleType) -> None:
    """Nula i číslo za počtem značek jsou navíc."""
    problems = markers.placeholder_problems("[[0]] [[1]] [[3]]", 1)
    assert problems.extra == (0, 3)
    assert problems.missing == ()


def test_problems_duplicated(markers: ModuleType) -> None:
    """Číslo použité dvakrát je zdvojené."""
    problems = markers.placeholder_problems("[[1]] [[1]] [[2]]", 2)
    assert problems.duplicated == (1,)
    assert problems.missing == ()


def test_problems_note_without_markers(markers: ModuleType) -> None:
    """Zpráva bez značek nemá žádný problém."""
    assert markers.placeholder_problems("text", 0) == markers.PlaceholderProblems(
        (), (), ()
    )


def test_problems_extra_numbers_beyond_expected(markers: ModuleType) -> None:
    """Čísla 0 a 2 při jedné očekávané značce jsou navíc."""
    problems = markers.placeholder_problems("[[0]] [[1]] [[2]]", 1)

    assert problems.extra == (0, 2)
    assert problems.missing == ()
