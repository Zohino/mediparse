# /// script
# requires-python = ">=3.13"
# ///
"""Maskuje značky ___ číslovanými [[n]] pro překlad a po překladu je vrací.

Číslování běží napříč celou zprávou, dělí se až maskovaný text. Číslo určuje,
která značka originálu se ztratila nebo přibyla. Pořadí čísel se neposuzuje,
český slovosled ho legitimně přehazuje.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import count
from typing import Final

PLACEHOLDER: Final = "[[n]]"
MARKER: Final = re.compile(r"(?<!_)___(?!_)")
NUMBERED: Final = re.compile(r"\[\[\s*(\d+)\s*\]\]")
BLANK: Final = "___"


@dataclass(frozen=True)
class PlaceholderProblems:
    """Čísla značek, která v překladu chybí, přebývají nebo jsou zdvojená."""

    missing: tuple[int, ...]
    extra: tuple[int, ...]
    duplicated: tuple[int, ...]


def mask(text: str) -> str:
    """Nahradí značky ___ v celé zprávě značkami [[n]] číslovanými od 1.

    Zpráva se dělí až po maskování, aby rezerva platila pro skutečný prompt.

    Args:
        text: Celá zpráva před dělením.

    Returns:
        Zpráva se značkami očíslovanými jedním čítačem napříč celým textem.
    """
    numbers = count(1)
    return MARKER.sub(lambda _: f"[[{next(numbers)}]]", text)


def restore(text: str) -> str:
    """Změní každou značku [[n]], i s mezerami uvnitř a s libovolným číslem, na ___.

    Returns:
        Text se značkami ___.
    """
    return NUMBERED.sub(BLANK, text)


def placeholder_problems(text: str, expected: int) -> PlaceholderProblems:
    """Porovná čísla značek v překladu s ``1..expected``.

    Args:
        text: Přeložený text se značkami [[n]].
        expected: Počet značek v originálu.

    Returns:
        Chybějící čísla, čísla mimo ``1..expected`` a čísla použitá vícekrát.
    """
    seen = Counter(int(number) for number in NUMBERED.findall(text))
    wanted = range(1, expected + 1)
    return PlaceholderProblems(
        missing=tuple(number for number in wanted if number not in seen),
        extra=tuple(sorted(number for number in seen if number not in wanted)),
        duplicated=tuple(number for number in wanted if seen[number] > 1),
    )
