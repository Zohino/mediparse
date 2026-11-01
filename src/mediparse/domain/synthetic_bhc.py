"""Nezávislý target BHC z pořadí sekcí plánu a celých řádků hlaviček."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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


def bhc_rows(
    plans: Sequence[NotePlan],
    discharge: Sequence[Mapping[str, str]],
    lexicon: SectionLexicon,
) -> list[dict[str, str]]:
    """Sestaví targety v pořadí note_id z dodaného textu zpráv.

    Returns:
        Řádky note_id a target; varianta řezu je pořadí mezi zprávami s BHC.

    Raises:
        InvalidInputError: Zpráva nemá plán, text postrádá hlavičku slíbenou plánem
            nebo má prázdný target.
    """
    by_note = {plan.note_id: plan for plan in plans}
    texts = {row["note_id"]: row["text"] for row in discharge}
    if without_plan := texts.keys() - by_note.keys():
        msg = f"Zprávy bez plánu: {', '.join(sorted(without_plan))}."
        raise InvalidInputError(msg)
    targets = []
    for note_id in sorted(texts):
        target = bhc_target(by_note[note_id], texts[note_id], lexicon, len(targets))
        if target is not None:
            targets.append({"note_id": note_id, "target": target})
    return targets


def bhc_target(
    plan: NotePlan, text: str, lexicon: SectionLexicon, variant: int
) -> str | None:
    """Vyřízne target podle celých řádků hlaviček slíbených plánem.

    Target nepoužívá parser sekcí: sešit 9 proti němu parser ověřuje.

    Returns:
        Text se sloučenými bílými znaky, nebo None pro plán bez BHC.

    Raises:
        InvalidInputError: Sekce plánu chybí v lexikonu, její hlavička v textu
            nebo je výsledný target prázdný.
    """
    if BHC_SECTION not in plan.sections:
        return None
    lines = text.splitlines()
    index = plan.sections.index(BHC_SECTION)
    following = plan.sections[index + 1 :]
    start = _header_line(lines, BHC_SECTION, lexicon) + 1
    end = _header_line(lines, following[0], lexicon, start) if following else len(lines)
    body = lines[start:end]
    next_end = (
        _header_line(lines, following[1], lexicon, end + 1)
        if len(following) > 1
        else len(lines)
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
    target = " ".join(" ".join(pieces).split())
    if not target:
        msg = f"Prázdný target BHC pro zprávu {plan.note_id}."
        raise InvalidInputError(msg)
    return target


def _header_line(
    lines: Sequence[str],
    section: str,
    lexicon: SectionLexicon,
    start: int = 0,
) -> int:
    if section not in lexicon.sections:
        msg = f"Lexikon nemá sekci plánu {section}."
        raise InvalidInputError(msg)
    headers = {f"{synonym}:" for synonym in lexicon.sections[section].synonyms}
    for index, line in enumerate(lines[start:], start=start):
        if line.strip() in headers:
            return index
    msg = f"Text nemá hlavičku sekce plánu {section}."
    raise InvalidInputError(msg)
