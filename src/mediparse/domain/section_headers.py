"""Lexikon sekcí propouštěcí zprávy a pravidlo, kterým se v textu pozná hlavička."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_NORMALIZED_CHARS: Final = 3
# Pravidlo hlavičky má tři části a každá má protějšek v eda/R/sections.R:
# HEADER_PATTERN, normalized (normalized_sql) a MIN_NORMALIZED_CHARS (MIN_HEADER_CHARS).
# Sešit 9 měřil pokrytí lexikonu právě jím, změna se dělá v obou. Délku hlídá validace
# lexikonu místo filtru: stejnost je rovnost normalizovaných tvarů, takže kratší tvar
# synonymem být nemůže a vyřazení kratších hlaviček z textu je tentýž výsledek.
HEADER_PATTERN: Final = re.compile(
    r"^[ \t]*(?P<header>[A-Z][A-Za-z][A-Za-z \-/&()'.]{0,47}):", re.MULTILINE
)


def normalized(header: str) -> str:
    """Převede hlavičku na tvar, v němž se porovnává se synonymy lexikonu.

    Returns:
        Malá písmena, běh bílých znaků jako jedna mezera, bez koncových mezer a teček.
    """
    return re.sub(r"\s+", " ", header.lower()).rstrip(" .")


class LexiconSection(BaseModel):
    """Sekce lexikonu: český popisek a synonyma jejího názvu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    label_cs: Annotated[str, Field(min_length=1)]
    synonyms: Annotated[tuple[str, ...], Field(min_length=1)]


class SectionLexicon(BaseModel):
    """Sekce zprávy v pořadí šablony a administrativní pole preambule.

    Synonyma jsou jen název sekce a jeho pravopisné varianty, aby žádná varianta
    neotevřela falešnou hranici.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sections: dict[str, LexiconSection]
    preamble: tuple[str, ...]

    @model_validator(mode="after")
    def _synonyms_are_recognizable_and_unique(self) -> Self:
        synonyms = [s for section in self.sections.values() for s in section.synonyms]
        for synonym in synonyms:
            _require_recognizable(synonym)
        forms = Counter(normalized(name) for name in [*synonyms, *self.preamble])
        clashes = sorted(form for form, count in forms.items() if count > 1)
        if clashes:
            msg = f"Po normalizaci splývají hlavičky: {', '.join(clashes)}."
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class SectionHeader:
    """Hlavička sekce nalezená v textu.

    Attributes:
        section: Klíč sekce v lexikonu.
        start: Offset začátku řádku hlavičky včetně odsazení.
        end: Offset za dvojtečkou.
    """

    section: str
    start: int
    end: int


def find_section_headers(
    text: str, lexicon: SectionLexicon
) -> tuple[SectionHeader, ...]:
    """Najde hlavičky sekcí lexikonu; hranicí je první výskyt každé sekce.

    Další výskyt téže sekce (i pod jiným synonymem) zůstává textem těla: sešit 9
    ukázal opakování hlavičky ve zprávách a opakovaný výskyt je falešná hranice.

    Returns:
        Hlavičky v pořadí textu.
    """
    section_of = {
        normalized(synonym): key
        for key, section in lexicon.sections.items()
        for synonym in section.synonyms
    }
    found: dict[str, SectionHeader] = {}
    for match in HEADER_PATTERN.finditer(text):
        key = section_of.get(normalized(match["header"]))
        if key is not None and key not in found:
            found[key] = SectionHeader(key, match.start(), match.end())
    return tuple(found.values())


def _require_recognizable(synonym: str) -> None:
    found = HEADER_PATTERN.fullmatch(f"{synonym}:")
    if found is None or found["header"] != synonym:
        msg = f"Synonymum {synonym!r} nesplňuje výraz hlavičky, parser by ho nenašel."
        raise ValueError(msg)
    if len(normalized(synonym)) < MIN_NORMALIZED_CHARS:
        msg = f"Synonymum {synonym!r} má normalizovaný tvar kratší než {MIN_NORMALIZED_CHARS} znaky."
        raise ValueError(msg)
