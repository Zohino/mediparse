"""Text syntetické propouštěcí zprávy: preambule, sekce lexikonu a target BHC.

Text vzniká z částí (hlavička a řádky těla), takže se z nich dá odvodit i to, co o zprávě
tvrdí jiné tabulky.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
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
        bhc_target: Target BHC odvozený z téhož textu.
    """

    text: str
    bhc_target: str


class _Cut(Enum):
    """Kde končí target BHC vzhledem k textu zprávy."""

    BODY = auto()
    THROUGH_NEXT_SECTION = auto()
    FIRST_LINE = auto()
    INTO_NEXT_SECTION = auto()


@dataclass(frozen=True)
class _Variant:
    """Rozdíl mezi bloky: co se do textu přidá nebo vynechá a kde končí target BHC."""

    cut: _Cut
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
    _Variant(_Cut.BODY),
    _Variant(_Cut.THROUGH_NEXT_SECTION),
    _Variant(_Cut.FIRST_LINE, repeats_diagnostic_header=True),
    _Variant(_Cut.INTO_NEXT_SECTION, skips_section_after_bhc=True),
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
            f"Za {BHC_SECTION} je sekcí: {after}, potřeba aspoň {_SECTIONS_AFTER_BHC}."
        )
        raise InvalidInputError(msg)


def synthetic_discharge(
    head: str,
    token: str,
    lexicon: SectionLexicon,
    block: int,
    mentions: tuple[str, ...] = (),
) -> SyntheticDischarge:
    """Složí text zprávy bloku a z téhož textu odvodí target BHC.

    Zpráva začíná hotovým prvním řádkem ``head`` (buňka s tokenem, který pro ni složí
    tabulky), pak má pole preambule se značkou ``___`` a všechny sekce lexikonu v jeho
    pořadí. Synonymum hlavičky se mění s blokem a tělo každé sekce nese token zprávy
    (DF = 1). Řádky ``mentions`` stojí na konci těla první sekce: ta není BHC ani sekce
    kolem něj, takže řezy targetu BHC a opakovaná hlavička varianty 2 zůstávají beze změny.

    Args:
        head: První řádek zprávy.
        token: Token zprávy ze samých písmen.
        lexicon: Lexikon sekcí z configu.
        block: Blok řádku.
        mentions: Hotové řádky zmínek klíčových slov, zkratek a příznaků.

    Returns:
        Text zprávy a target BHC.
    """
    variant = _VARIANTS[block % len(_VARIANTS)]
    parts = _parts(lexicon, block, token, variant)
    parts[0] = _Part(parts[0].key, parts[0].header, (*parts[0].lines, *mentions))
    preamble = [f"{field}: {DEID}" for field in lexicon.preamble]
    body = [line for part in parts for line in (part.header, *part.lines, "")]
    text = "\n".join([head, *preamble, *body]) + "\n"
    return SyntheticDischarge(text, _bhc_target(parts, variant.cut))


def _bhc_target(parts: list[_Part], cut: _Cut) -> str:
    """Složí target BHC z částí, které generátor vytvořil.

    Target se neodvozuje z ``split_sections``: sešit 9 parser proti targetu ověřuje, takže
    by se z konstrukce vždy sešel a chyba parseru by zůstala skrytá. Text úseku má sloučené bílé
    znaky jako po čištění datasetu MIMIC-IV-Ext-BHC.

    Returns:
        Text úseku za hlavičkou BHC.
    """
    index = next(i for i, part in enumerate(parts) if part.key == BHC_SECTION)
    bhc, following = parts[index], parts[index + 1]
    pieces = {
        _Cut.BODY: bhc.lines,
        _Cut.THROUGH_NEXT_SECTION: (*bhc.lines, following.header, *following.lines),
        _Cut.FIRST_LINE: bhc.lines[:1],
        _Cut.INTO_NEXT_SECTION: (*bhc.lines, following.header, following.lines[0]),
    }[cut]
    return " ".join(" ".join(pieces).split())


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
