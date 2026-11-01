"""Nezávislý target BHC z pořadí sekcí plánu a celých řádků hlaviček."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.section_headers import SectionLexicon

# Protějšek BHC_SECTION z eda/R/bhc.R; role BHC není součástí schématu lexikonu.
BHC_SECTION: Final = "hospital_course"


class _Cut(IntEnum):
    """Konec targetu vzhledem k tělu BHC a následující sekci."""

    BODY = 0
    THROUGH_NEXT_SECTION = 1
    FIRST_LINE = 2
    INTO_NEXT_SECTION = 3


def bhc_target(
    plan: NotePlan, text: str, lexicon: SectionLexicon, variant: int
) -> str | None:
    """Vyřízne target podle celých řádků hlaviček slíbených plánem.

    Target nepoužívá parser sekcí: sešit 9 proti němu parser ověřuje.

    Returns:
        Text se sloučenými bílými znaky, nebo None pro plán bez BHC.

    Raises:
        InvalidInputError: Sekce plánu chybí v lexikonu nebo její hlavička v textu.
    """
    if BHC_SECTION not in plan.sections:
        return None
    lines = text.splitlines()
    index = plan.sections.index(BHC_SECTION)
    following = plan.sections[index + 1 :]
    start = _header_line(lines, BHC_SECTION, lexicon) + 1
    end = _header_line(lines, following[0], lexicon) if following else len(lines)
    body = lines[start:end]
    next_end = (
        _header_line(lines, following[1], lexicon) if len(following) > 1 else len(lines)
    )
    next_lines = lines[end:next_end] if following else []
    cut = _Cut(variant % len(_Cut))
    if cut is _Cut.FIRST_LINE:
        pieces = [line for line in body if line.strip()][:1]
    elif cut is _Cut.THROUGH_NEXT_SECTION:
        pieces = [*body, *next_lines]
    elif cut is _Cut.INTO_NEXT_SECTION:
        pieces = [*body, *[line for line in next_lines if line.strip()][:2]]
    else:
        pieces = body
    return " ".join(" ".join(pieces).split())


def _header_line(lines: Sequence[str], section: str, lexicon: SectionLexicon) -> int:
    if section not in lexicon.sections:
        msg = f"Lexikon nemá sekci plánu {section}."
        raise InvalidInputError(msg)
    headers = {f"{synonym}:" for synonym in lexicon.sections[section].synonyms}
    for index, line in enumerate(lines):
        if line.strip() in headers:
            return index
    msg = f"Text nemá hlavičku sekce plánu {section}."
    raise InvalidInputError(msg)
