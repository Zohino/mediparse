"""Text syntetické propouštěcí zprávy: preambule, sekce lexikonu a token zprávy.

Text vzniká z částí (hlavička a řádky těla), takže se z nich dá odvodit i to, co o zprávě
tvrdí jiné tabulky.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from string import ascii_lowercase
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note_text import DEID

if TYPE_CHECKING:
    from mediparse.domain.section_headers import SectionLexicon

# Protějšek BHC_SECTION z eda/R/bhc.R. Lexikon (config/section_headers.json) čtou R i Python,
# ale role BHC není součástí jeho schématu a kvůli dočasnému propojení obou jazyků (S20h11
# R odstraní) se schéma nerozšiřuje. Změna klíče se dělá v obou souborech.
BHC_SECTION: Final = "hospital_course"
# Za BHC musí být dvě sekce: varianta 3 jednu vynechá a target řeže do následující.
_SECTIONS_AFTER_BHC: Final = 2


@dataclass(frozen=True)
class SyntheticDischarge:
    """Syntetická propouštěcí zpráva a target MIMIC-IV-Ext-BHC odvozený z téhož textu.

    Attributes:
        text: Text zprávy.
        bhc_target: Target BHC, nebo None u zprávy, jejíž target zůstává buňkou.
    """

    text: str
    bhc_target: str | None


@dataclass(frozen=True)
class _Variant:
    """Rozdíl mezi bloky: co se do textu přidá nebo vynechá a kde končí target BHC."""

    skips_section_after_bhc: bool = False
    repeats_diagnostic_header: bool = False


@dataclass(frozen=True)
class _Part:
    """Sekce zprávy: klíč lexikonu, řádek hlavičky a řádky těla."""

    key: str
    header: str
    lines: tuple[str, ...]


# Blok se na variantu mapuje modulo, aby každá varianta měla aspoň MIN_CELL zpráv
# i při jiném počtu bloků, než jaký tabulky dnes mají.
_VARIANTS: Final = (
    _Variant(),
    _Variant(),
    _Variant(repeats_diagnostic_header=True),
    _Variant(skips_section_after_bhc=True),
)


def require_bhc_layout(lexicon: SectionLexicon) -> None:
    """Ověří, že lexikon má sekci BHC a za ní aspoň dvě sekce.

    Raises:
        InvalidInputError: Lexikon nemá klíč ``hospital_course`` nebo za ním nejsou dvě sekce.
    """
    keys = list(lexicon.sections)
    if BHC_SECTION not in keys:
        msg = f"Lexikon sekcí nemá sekci {BHC_SECTION}, target BHC by neměl z čeho vzniknout."
        raise InvalidInputError(msg)
    after = len(keys) - keys.index(BHC_SECTION) - 1
    if after < _SECTIONS_AFTER_BHC:
        msg = (
            f"Lexikon sekcí má za {BHC_SECTION} jen {after} sekcí, "
            f"potřebuje aspoň {_SECTIONS_AFTER_BHC}."
        )
        raise InvalidInputError(msg)


def synthetic_discharge(
    prefix: str,
    lexicon: SectionLexicon,
    row: int,
    block: int | None,
    text_chars: int,
) -> SyntheticDischarge:
    """Složí text zprávy řádku ``row``.

    Zpráva bloku má buňku na prvním řádku, pole preambule se značkou ``___`` a všechny sekce
    lexikonu v jeho pořadí. Synonymum hlavičky se mění s blokem a tělo každé sekce nese
    token zprávy (DF = 1). Zpráva bez bloku (vzácný řádek) je buňka a token do ``text_chars``
    znaků, bez nadpisů.

    Args:
        prefix: Prefix buněk a tokenů.
        lexicon: Lexikon sekcí z configu.
        row: Řádek tabulky.
        block: Blok řádku, nebo None pro vzácný řádek.
        text_chars: Nejmenší délka textu vzácného řádku.

    Returns:
        Text a target BHC (u vzácného řádku None).
    """
    cell = discharge_cell(prefix, row)
    if block is None:
        return SyntheticDischarge(
            padded_text(cell, text_token(prefix, row), text_chars), None
        )
    variant = _VARIANTS[block % len(_VARIANTS)]
    parts = _parts(lexicon, block, text_token(prefix, row), variant)
    preamble = [f"{field}: {DEID}" for field in lexicon.preamble]
    body = [line for part in parts for line in (part.header, *part.lines, "")]
    return SyntheticDischarge("\n".join([cell, *preamble, *body]) + "\n", None)


def discharge_cell(prefix: str, row: int) -> str:
    """Vrátí buňku textu propouštěcí zprávy.

    Returns:
        Buňka ve tvaru ``prefix-discharge-text-řádek``.
    """
    return f"{prefix}-discharge-text-{row}"


def padded_text(cell: str, token: str, text_chars: int) -> str:
    """Připojí za buňku token, dokud text nemá aspoň ``text_chars`` znaků.

    Returns:
        Buňka a opakovaný token oddělené mezerou.
    """
    copies = max(1, math.ceil((text_chars - len(cell)) / (len(token) + 1)))
    return " ".join([cell, *[token] * copies])


def _parts(
    lexicon: SectionLexicon, block: int, token: str, variant: _Variant
) -> list[_Part]:
    keys = list(lexicon.sections)
    skipped = (
        keys[keys.index(BHC_SECTION) + 1] if variant.skips_section_after_bhc else None
    )
    parts = []
    for key, section in lexicon.sections.items():
        if key == skipped:
            continue
        synonym = section.synonyms[block % len(section.synonyms)]
        # Pseudohlavička ``Token:`` není chyba formátu: je to popisek z dat s n = 1, který
        # žebříček hlaviček v sešitu 9 musí skrýt, a když selže, kanárek ho najde.
        lines = (f"{token.capitalize()}:", f"{synonym.lower()} {token}")
        parts.append(_Part(key, f"{synonym}:", lines))
    if variant.repeats_diagnostic_header:
        parts[-1] = _with_repeated_header(parts[-1], lexicon, block)
    return parts


def _with_repeated_header(part: _Part, lexicon: SectionLexicon, block: int) -> _Part:
    diagnostic = next(
        section for section in lexicon.sections.values() if section.diagnostic
    )
    synonym = diagnostic.synonyms[block % len(diagnostic.synonyms)]
    return _Part(part.key, part.header, (*part.lines, f"{synonym}:"))


def text_token(prefix: str, row: int) -> str:
    """Vrátí token zprávy: písmena prefixu a číslo řádku zapsané písmeny.

    Tokenizace zahodí pomlčky a n-gramy EDA berou jen písmena (``[a-z]{2,}``), takže buňku
    s pomlčkami a číslem řádku ve výstupech po tokenech (n-gramy, slovník) kanárek nenajde.
    Token ze samých písmen projde beze změny a v každé zprávě je jiný (DF = 1); s číslicí by
    se v n-gramech zkrátil na kmen, který mají všechny zprávy. Výstup, ve kterém se token
    objeví, proto pustil token jediné zprávy.

    Returns:
        Token ze samých písmen.
    """
    stem = "".join(character for character in prefix if character.isalpha())
    return stem + _letters(row)


def _letters(number: int) -> str:
    """Zapíše nezáporné číslo písmeny jako číslice soustavy o 26 znacích.

    Returns:
        Číslo zapsané písmeny, například 0 → a, 25 → z, 26 → ba.
    """
    digits = ""
    while True:
        number, digit = divmod(number, len(ascii_lowercase))
        digits = ascii_lowercase[digit] + digits
        if number == 0:
            return digits
