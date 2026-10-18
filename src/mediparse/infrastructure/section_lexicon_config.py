"""Načtení lexikonu sekcí z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.section_headers import SectionLexicon
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_section_lexicon(path: Path) -> SectionLexicon:
    """Přečte JSON lexikon a předá ho validaci.

    Returns:
        Lexikon, který prošel schématem.
    """
    return parse_file(path, SectionLexicon.model_validate_json)
