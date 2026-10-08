"""Kontroly českého překladu syntetického korpusu proti anglickým originálům."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

LENGTH_RATIO: Final = (0.8, 1.6)
_MARKER: Final = re.compile(r"(?<!_)___(?!_)")


def deid_markers(text: str) -> int:
    """Počet značek ___ nahrazujících vyřazené identifikátory.

    Returns:
        Počet řad právě tří podtržítek; delší ani kratší řady se nepočítají.
    """
    return len(_MARKER.findall(text))


def translation_violations(
    english: Mapping[str, str], czech: Mapping[str, str]
) -> tuple[str, ...]:
    """Důvody, proč český korpus není úplným překladem anglického.

    Args:
        english: Anglické texty klíčované note_id.
        czech: České texty klíčované note_id.

    Returns:
        Porušení seřazená podle note_id; bez českých textů nic, protože korpus před
        překladem nic neporušuje.
    """
    if not czech:
        return ()
    return tuple(
        reason
        for note_id in sorted(english.keys() | czech.keys())
        for reason in _reasons(note_id, english.get(note_id), czech.get(note_id))
    )


def marker_mismatches(
    english: Mapping[str, str], czech: Mapping[str, str]
) -> tuple[str, ...]:
    """Zprávy, jejichž překlad má jiný počet značek ___ než originál.

    Returns:
        Seřazená note_id zpráv, které mají originál i překlad.
    """
    return tuple(
        note_id
        for note_id in sorted(english.keys() & czech.keys())
        if deid_markers(english[note_id]) != deid_markers(czech[note_id])
    )


def _reasons(note_id: str, english: str | None, czech: str | None) -> tuple[str, ...]:
    if czech is None:
        return (f"cs/{note_id}: chybí překlad anglické zprávy.",)
    if english is None:
        return (f"cs/{note_id}: překlad nemá anglický originál.",)
    if not czech.strip():
        return (f"cs/{note_id}: překlad je prázdný.",)
    if not english.strip():
        return (f"en/{note_id}: anglický originál je prázdný.",)
    low, high = LENGTH_RATIO
    ratio = len(czech) / len(english)
    if low <= ratio <= high:
        return ()
    reason = f"cs/{note_id}: poměr délky {ratio:.2f} leží mimo {low} až {high}."
    return (reason,)
