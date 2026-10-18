"""Lexikon sekcí propouštěcí zprávy, rozpoznání hlaviček v textu a rozdělení zprávy na sekce."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_NORMALIZED_CHARS: Final = 3
# Klíč pro text mimo rozpoznané sekce; v lexikonu být nesmí, jinak by splynul se zbytkem.
OTHER: Final = "other"
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

    sections: Annotated[dict[str, LexiconSection], Field(min_length=1)]
    preamble: tuple[str, ...]

    @model_validator(mode="after")
    def _reserved_key_is_absent(self) -> Self:
        if OTHER in self.sections:
            msg = f"Sekce lexikonu nesmí mít vyhrazený klíč: {OTHER}."
            raise ValueError(msg)
        return self

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


@dataclass(frozen=True)
class NoteSection:
    """Jednotka rozdělené zprávy: hlavička sekce a její tělo.

    Hlavička zahrnuje odsazení před názvem, protože `SectionHeader.start` je začátek
    řádku. Konec řádku za dvojtečkou patří do těla, takže odstranění hlavičky i těla
    nezanechá prázdný řádek navíc.

    Attributes:
        section: Klíč sekce v lexikonu, nebo `OTHER` pro text mimo sekce.
        header: Text od začátku řádku po dvojtečku včetně; u `OTHER` prázdný.
        body: Text do začátku další hlavičky.
    """

    section: str
    header: str
    body: str


def split_sections(text: str, lexicon: SectionLexicon) -> tuple[NoteSection, ...]:
    """Rozdělí zprávu na sekce v pořadí textu; text před první hlavičkou je `OTHER`.

    Neprázdný `OTHER` stojí na začátku (zpráva bez hlavičky je jím celá). Spojení
    `header + body` všech sekcí dává přesně původní text.

    Returns:
        Sekce v pořadí textu, pro prázdný text prázdná n-tice.
    """
    headers = find_section_headers(text, lexicon)
    starts = [h.start for h in headers] + [len(text)]
    preamble = text[: starts[0]]
    sections = [NoteSection(OTHER, "", preamble)] if preamble else []
    sections.extend(
        NoteSection(h.section, text[h.start : h.end], text[h.end : end])
        for h, end in zip(headers, starts[1:], strict=True)
    )
    return tuple(sections)


def _require_recognizable(synonym: str) -> None:
    found = HEADER_PATTERN.fullmatch(f"{synonym}:")
    if found is None or found["header"] != synonym:
        msg = f"Synonymum {synonym!r} nesplňuje výraz hlavičky, parser by ho nenašel."
        raise ValueError(msg)
    if len(normalized(synonym)) < MIN_NORMALIZED_CHARS:
        msg = f"Synonymum {synonym!r} má normalizovaný tvar kratší než {MIN_NORMALIZED_CHARS} znaky."
        raise ValueError(msg)
