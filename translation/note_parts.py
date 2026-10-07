# /// script
# requires-python = ">=3.13"
# ///
"""Dělí zprávu na části, které se vejdou do okna překladu, a skládá je zpátky.

Jednotkou dělení je blok oddělený prázdným řádkem, záložní jednotkou řádek.
Modul nezná tokenizér, vejde-li se text, určuje predikát od volajícího.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

BLOCK_SEPARATOR: Final = "\n\n"
LINE_SEPARATOR: Final = "\n"


class UnsplittableError(ValueError):
    """Řádek zprávy se nevejde do okna ani sám."""


def split(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Rozdělí text na části, z nichž každá splní ``fits``.

    Když se nevejde ani jeden řádek, vyvolá ``UnsplittableError``.

    Args:
        text: Celá zpráva.
        fits: Predikát, zda se text vejde do okna i s rezervou na překlad.

    Returns:
        ``[text]`` beze změny, když se vejde celý, jinak hladově zbalené části.
    """
    if fits(text):
        return [text]
    parts: list[str] = []
    current: list[str] = []
    for block in _units(text, BLOCK_SEPARATOR):
        if fits(BLOCK_SEPARATOR.join([*current, block])):
            current.append(block)
            continue
        if current:
            parts.append(BLOCK_SEPARATOR.join(current))
        current = []
        if fits(block):
            current.append(block)
        else:
            parts.extend(_pack(_units(block, LINE_SEPARATOR), LINE_SEPARATOR, fits))
    if current:
        parts.append(BLOCK_SEPARATOR.join(current))
    return parts


def join(parts: Sequence[str]) -> str:
    """Spojí přeložené části do jedné zprávy.

    Args:
        parts: Části v pořadí zprávy.

    Returns:
        Části bez okrajových mezer oddělené prázdným řádkem.
    """
    return BLOCK_SEPARATOR.join(part.strip() for part in parts)


def part_ids(note_id: str, count: int) -> list[str]:
    """Určí ID požadavků pro části zprávy.

    Args:
        note_id: ID zprávy.
        count: Počet částí.

    Returns:
        ``[note_id]`` pro jednu část, jinak ``note_id/1`` až ``note_id/count``.
    """
    if count == 1:
        return [note_id]
    return [f"{note_id}/{number}" for number in range(1, count + 1)]


def _units(text: str, separator: str) -> list[str]:
    return [unit for unit in map(str.strip, text.split(separator)) if unit]


def _pack(
    lines: Sequence[str], separator: str, fits: Callable[[str], bool]
) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for line in lines:
        if not fits(line):
            msg = f"Řádek o {len(line)} znacích se nevejde do okna."
            raise UnsplittableError(msg)
        if current and not fits(separator.join([*current, line])):
            parts.append(separator.join(current))
            current = []
        current.append(line)
    if current:
        parts.append(separator.join(current))
    return parts
