"""Délky, příznaky a offsety částí zprávy podle parseru sekcí."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.note_text import DEID
from mediparse.domain.section_headers import split_sections

if TYPE_CHECKING:
    from mediparse.domain.section_headers import SectionLexicon


@dataclass(frozen=True)
class SectionSpan:
    """Část zprávy s offsety těla v code pointech před ořezem bílých znaků."""

    note_id: str
    position: int
    section: str
    span_chars: int
    body_chars: int
    body_masked: bool
    body_start: int
    body_end: int


def section_spans(
    note_id: str, text: str, lexicon: SectionLexicon
) -> list[SectionSpan]:
    """Rozdělí zprávu a změří části v pořadí textu včetně částí mimo sekce.

    Returns:
        Délky celé části a oříznutého těla, příznak masky a offsety těla.
    """
    rows = []
    cursor = 0
    for position, part in enumerate(split_sections(text, lexicon)):
        body_start = cursor + len(part.header)
        cursor = body_start + len(part.body)
        body = part.body.strip()
        rows.append(
            SectionSpan(
                note_id,
                position,
                part.section,
                len(part.header) + len(part.body),
                len(body),
                body == DEID,
                body_start,
                cursor,
            )
        )
    return rows
