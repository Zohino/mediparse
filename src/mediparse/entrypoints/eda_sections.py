"""Rozdělení zpráv na sekce pro sešit EDA, který modul volá přes reticulate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mediparse.domain.section_headers import OTHER as _OTHER
from mediparse.domain.section_headers import split_sections
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

OTHER: Final = _OTHER


def note_sections(
    lexicon: Path, note_ids: Sequence[str], texts: Sequence[str]
) -> dict[str, list[str] | list[int]]:
    """Rozdělí zprávy parserem sekcí a vrátí délky po sekcích jako sloupce.

    Text se přes most nevrací, sešit potřebuje jen pozice a délky. Lexikon se načítá při
    každém volání, aby funkce zůstala bezstavová (soubor má desítky řádků).

    Args:
        lexicon: Cesta k lexikonu sekcí.
        note_ids: Identifikátory zpráv.
        texts: Texty zpráv ve stejném pořadí jako ``note_ids``.

    Returns:
        Sloupce ``note_id``, ``position`` (pořadí v textu od 0 včetně ``other``),
        ``section``, ``span_chars`` (hlavička i tělo) a ``body_chars`` (tělo bez
        okrajových bílých znaků), vždy všech pět, i pro prázdný vstup.

    Raises:
        ValueError: ``note_ids`` a ``texts`` mají různou délku.
        InvalidInputError: Lexikon chybí nebo neodpovídá schématu.
    """
    parsed = load_section_lexicon(lexicon)
    ids: list[str] = []
    positions: list[int] = []
    sections: list[str] = []
    spans: list[int] = []
    bodies: list[int] = []
    for note_id, text in zip(note_ids, texts, strict=True):
        for position, part in enumerate(split_sections(text, parsed)):
            ids.append(note_id)
            positions.append(position)
            sections.append(part.section)
            spans.append(len(part.header) + len(part.body))
            bodies.append(len(part.body.strip()))
    return {
        "note_id": ids,
        "position": positions,
        "section": sections,
        "span_chars": spans,
        "body_chars": bodies,
    }
