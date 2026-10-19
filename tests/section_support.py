"""Lexikon sekcí z config/ a zprávy ve tvaru šablony MIMIC pro testy parseru sekcí."""

from __future__ import annotations

from typing import Final

from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from tests.support import REPOSITORY, REPOSITORY_SECTION_HEADERS

REPOSITORY_LEXICON: Final = load_section_lexicon(REPOSITORY_SECTION_HEADERS)
MIMIC_SHAPED_NOTES: Final = REPOSITORY / "tests" / "fixtures" / "mimic_shaped"


def mimic_shaped_notes() -> dict[str, str]:
    """Zprávy ve tvaru šablony MIMIC načtené beze změny konců řádků.

    Returns:
        Jméno souboru bez přípony a text, seřazené podle jména.
    """
    return {
        path.stem: path.read_bytes().decode("utf-8")
        for path in sorted(MIMIC_SHAPED_NOTES.glob("*.txt"))
    }
