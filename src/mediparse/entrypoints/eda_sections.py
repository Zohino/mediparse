"""Rozdělení zpráv na sekce pro sešit EDA, který modul volá přes reticulate."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Final

from mediparse.domain.note_sections import section_spans
from mediparse.domain.section_headers import OTHER as _OTHER
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

OTHER: Final = _OTHER


def note_sections(
    lexicon: Path, note_ids: Sequence[str], texts: Sequence[str]
) -> dict[str, list[str] | list[int] | list[bool]]:
    """Rozdělí zprávy parserem sekcí a vrátí délky po sekcích jako sloupce.

    Text se přes most nevrací: sešit dostane délky a offsety těla a text si vyřízne sám
    z tabulky zpráv, takže pravidlo hranice sekce zůstává jen v parseru. Lexikon se načítá
    při každém volání, aby funkce zůstala bezstavová (soubor má desítky řádků).

    Args:
        lexicon: Cesta k lexikonu sekcí.
        note_ids: Identifikátory zpráv.
        texts: Texty zpráv ve stejném pořadí jako ``note_ids``.

    Returns:
        Sloupce ``note_id``, ``position`` (pořadí v textu od 0 včetně ``other``),
        ``section``, ``span_chars`` (hlavička i tělo), ``body_chars`` (tělo bez
        okrajových bílých znaků) a ``body_masked`` (tělo je jen maska
        de-identifikace), ``body_start`` a ``body_end`` (offsety těla v textu zprávy
        v code pointech, ``text[body_start:body_end]`` je tělo před ořezem), vždy
        všechny, i pro prázdný vstup.

    Raises:
        ValueError: ``note_ids`` a ``texts`` mají různou délku.
        InvalidInputError: Lexikon chybí nebo neodpovídá schématu.
    """
    parsed = load_section_lexicon(lexicon)
    columns: dict[str, list] = {
        name: []
        for name in (
            "note_id",
            "position",
            "section",
            "span_chars",
            "body_chars",
            "body_masked",
            "body_start",
            "body_end",
        )
    }
    for note_id, text in zip(note_ids, texts, strict=True):
        for row in section_spans(note_id, text, parsed):
            for name, value in asdict(row).items():
                columns[name].append(value)
    return columns
