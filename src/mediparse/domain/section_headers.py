"""Lexikon sekcí propouštěcí zprávy, rozpoznání hlaviček v textu, rozdělení zprávy na sekce a strip diagnostických sekcí."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mediparse.domain.language import Language

if TYPE_CHECKING:
    from collections.abc import Mapping

MIN_NORMALIZED_CHARS: Final = 3
# Hlavičky a synonyma v config/section_headers.json jsou anglické (šablona MIMIC-IV-Note),
# strip a validace matice proto smí pracovat jen se zprávami v tomto jazyce.
LEXICON_LANGUAGE: Final = Language.EN
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
    """Sekce lexikonu: český popisek, synonyma názvu a příznak diagnostické sekce.

    Attributes:
        label_cs: Český popisek sekce.
        synonyms: Název sekce a jeho pravopisné varianty.
        diagnostic: Sekce, ze které se kódují diagnózy a kterou ablace D2 odstraní.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    label_cs: Annotated[str, Field(min_length=1)]
    synonyms: Annotated[tuple[str, ...], Field(min_length=1)]
    diagnostic: bool = False


class SectionLexicon(BaseModel):
    """Sekce zprávy v pořadí šablony a administrativní pole preambule.

    Synonyma jsou jen název sekce a jeho pravopisné varianty, aby žádná varianta
    neotevřela falešnou hranici.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sections: Annotated[dict[str, LexiconSection], Field(min_length=1)]
    preamble: tuple[str, ...]

    @property
    def diagnostic_sections(self) -> frozenset[str]:
        """Klíče sekcí označených jako diagnostické.

        Returns:
            Klíče, které odstraní `strip_diagnostic_sections`.
        """
        return frozenset(
            key for key, section in self.sections.items() if section.diagnostic
        )

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
            _require_recognizable(synonym, "Synonymum")
        # Pole preambule ukončují sekci (split_sections), nerozpoznatelné by tiše nikdy nezačalo.
        for name in self.preamble:
            _require_recognizable(name, "Pole")
        forms = Counter(normalized(name) for name in [*synonyms, *self.preamble])
        clashes = sorted(form for form, count in forms.items() if count > 1)
        if clashes:
            msg = f"Po normalizaci splývají hlavičky: {', '.join(clashes)}."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _has_diagnostic_section(self) -> Self:
        # Bez diagnostické sekce by strip tiše nic neodstranil a ablace by se rovnala hlavnímu řádku.
        if not self.diagnostic_sections:
            msg = "Lexikon musí mít aspoň jednu diagnostickou sekci (diagnostic: true)."
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
    return tuple(
        SectionHeader(key, match.start(), match.end())
        for key, match in _first_matches(text, section_of).items()
    )


@dataclass(frozen=True)
class NoteSection:
    """Jednotka rozdělené zprávy: hlavička sekce a její tělo.

    Hlavička zahrnuje odsazení před názvem, protože `SectionHeader.start` je začátek
    řádku. Konec řádku za dvojtečkou patří do těla, takže odstranění hlavičky i těla
    nezanechá prázdný řádek navíc.

    Attributes:
        section: Klíč sekce v lexikonu, nebo `OTHER` pro text mimo sekce.
        header: Text od začátku řádku po dvojtečku včetně; u `OTHER` prázdný.
        body: Text do další hranice, tedy hlavičky sekce nebo administrativního pole.
    """

    section: str
    header: str
    body: str


def split_sections(text: str, lexicon: SectionLexicon) -> tuple[NoteSection, ...]:
    """Rozdělí zprávu na sekce v pořadí textu; text mimo sekce je `OTHER`.

    `OTHER` je text před první hlavičkou a text od administrativního pole lexikonu
    (`preamble`) do další hlavičky: šablona MIMIC má pole Attending za Allergies a bez
    hranice by leželo v těle alergií. Hranicí je první výskyt pole na začátku řádku
    (i v preambuli), opakování je text těla jako u sekcí. Sousední `OTHER` se slijí,
    takže zpráva bez hlavičky je jedinou částí `OTHER`. Spojení `header + body` všech
    částí dává přesně původní text.

    Returns:
        Části v pořadí textu, pro prázdný text prázdná n-tice.
    """
    headers = {h.start: h for h in find_section_headers(text, lexicon)}
    fields = {normalized(name): name for name in lexicon.preamble}
    cuts = [match.start() for match in _first_matches(text, fields).values()]
    sections: list[NoteSection] = []
    for start, end in pairwise([*sorted({0, *headers, *cuts}), len(text)]):
        header = headers.get(start)
        if header is not None:
            sections.append(
                NoteSection(
                    header.section, text[start : header.end], text[header.end : end]
                )
            )
        elif sections and sections[-1].section == OTHER:
            sections[-1] = replace(
                sections[-1], body=sections[-1].body + text[start:end]
            )
        elif start < end:
            sections.append(NoteSection(OTHER, "", text[start:end]))
    return tuple(sections)


def strip_diagnostic_sections(text: str, lexicon: SectionLexicon) -> str:
    """Odstraní ze zprávy sekce lexikonu označené jako diagnostické, i s hlavičkou.

    Hranici sekce určuje jen `split_sections`, takže zbytek je po bajtech původní text.
    Opakovaná hlavička v těle jiné sekce zůstává (hranicí je první výskyt), proto strip
    není idempotentní.

    Returns:
        Text bez diagnostických sekcí.
    """
    return "".join(
        part.header + part.body
        for part in split_sections(text, lexicon)
        if part.section not in lexicon.diagnostic_sections
    )


def _first_matches(text: str, key_of: Mapping[str, str]) -> dict[str, re.Match[str]]:
    """Najde první výskyt každého klíče mezi hlavičkami podle `HEADER_PATTERN`.

    Returns:
        Klíč z `key_of` (podle normalizovaného tvaru hlavičky) a jeho první nález.
    """
    found: dict[str, re.Match[str]] = {}
    for match in HEADER_PATTERN.finditer(text):
        key = key_of.get(normalized(match["header"]))
        if key is not None and key not in found:
            found[key] = match
    return found


def _require_recognizable(name: str, kind: str) -> None:
    found = HEADER_PATTERN.fullmatch(f"{name}:")
    if found is None or found["header"] != name:
        msg = f"{kind} {name!r} nesplňuje výraz hlavičky, parser by ho nenašel."
        raise ValueError(msg)
    if len(normalized(name)) < MIN_NORMALIZED_CHARS:
        msg = f"{kind} {name!r} má normalizovaný tvar kratší než {MIN_NORMALIZED_CHARS} znaky."
        raise ValueError(msg)
