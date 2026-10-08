"""Kontroly českého překladu syntetického korpusu proti anglickým originálům."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

LENGTH_RATIO: Final = (0.8, 1.6)
_MARKER: Final = re.compile(r"(?<!_)___(?!_)")


@dataclass(frozen=True)
class TranslatedNote:
    """Přeložená zpráva: její note_id, otisk originálu v době přípravy a přeložené části v pořadí."""

    note_id: str
    source_sha256: str
    parts: tuple[str, ...]


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


def joined_translation(parts: Sequence[str]) -> str:
    """Složí přeložené části do zprávy tak, jak ji zapisuje sběr překladů.

    Returns:
        Části bez okrajových mezer oddělené prázdným řádkem; řádky bez koncových
        mezer, text končí novým řádkem.
    """
    text = "\n\n".join(part.strip() for part in parts)
    return "".join(f"{line.rstrip()}\n" for line in text.strip().splitlines())


def binding_violations(
    notes: Iterable[TranslatedNote],
    english: Mapping[str, str],
    czech: Mapping[str, str],
) -> tuple[str, ...]:
    """Důvody, proč překlad v pracovním adresáři nepatří k českému a anglickému korpusu.

    Args:
        notes: Zprávy z požadavků a výstupů překladu.
        english: Anglické texty korpusu klíčované note_id.
        czech: České texty korpusu klíčované note_id.

    Returns:
        Zprávy s jiným originálem, note_id mimo požadavky či korpus a české texty,
        které nejsou složením výstupů; seřazeno podle note_id.
    """
    by_id = {note.note_id: note for note in notes}
    reasons = [
        f"cs/{note_id}: chybí v cs/." for note_id in sorted(by_id.keys() - czech.keys())
    ]
    reasons.extend(
        f"cs/{note_id}: není v požadavcích překladu."
        for note_id in sorted(czech.keys() - by_id.keys())
    )
    for note_id in sorted(by_id.keys() & czech.keys()):
        note = by_id[note_id]
        if _sha256(english.get(note_id)) != note.source_sha256:
            reasons.append(f"en/{note_id}: originál se od požadavku změnil.")
        if czech[note_id] != joined_translation(note.parts):
            reasons.append(f"cs/{note_id}: text není složením výstupů překladu.")
    return tuple(reasons)


def _sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode()).hexdigest()
