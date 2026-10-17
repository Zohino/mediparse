"""Text zprávy: de-identifikační značka a její počet, sekce podle hlaviček na začátku řádku, věková značka a měření narativu."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.note_structure import StructureModel

DEID: Final = "___"
# Delší řady podtržítek nejsou značka (hlásí je kontrola cizí formy značky). Vzor
# proto nelze zjednodušit na str.count, ten by ____ započítal jako značku.
STANDALONE_DEID: Final = rf"(?<!_){re.escape(DEID)}(?!_)"
_DEID_PATTERN: Final = re.compile(STANDALONE_DEID)
AGE_SUFFIX: Final = r"[ -]?(?:years?[ -]old|y/?o)\b"
AGE_MARKER: Final = re.compile(rf"{STANDALONE_DEID}{AGE_SUFFIX}", re.IGNORECASE)

type Sections = tuple[tuple[str, str], ...]


def deid_markers(text: str) -> int:
    """Počet značek ___ nahrazujících vyřazené identifikátory.

    Returns:
        Počet řad právě tří podtržítek; delší ani kratší řady se nepočítají.
    """
    return len(_DEID_PATTERN.findall(text))


def alternation(words: Iterable[str]) -> str:
    """Alternace regulárního výrazu, v níž delší slovo vyhrává nad svou předponou.

    Returns:
        Escapovaná slova od nejdelšího spojená svislou čarou.
    """
    return "|".join(map(re.escape, sorted(words, key=len, reverse=True)))


def line_headers(headers: Iterable[str]) -> re.Pattern[str]:
    """Hlavičky s dvojtečkou na začátku řádku.

    Returns:
        Zkompilovaný vzor, skupina ``header`` nese nalezenou hlavičku.
    """
    return re.compile(rf"^(?P<header>{alternation(headers)}):", re.MULTILINE)


def segment(text: str, headers: Mapping[str, str]) -> tuple[str, Sections]:
    """Rozdělí text na preambuli a sekce podle kanonických hlaviček na začátku řádku.

    Returns:
        Preambuli a dvojice klíč sekce, tělo sekce v pořadí textu.
    """
    keys = {header: key for key, header in headers.items()}
    matches = list(line_headers(keys).finditer(text))
    ends = [match.start() for match in matches[1:]] + [len(text)]
    sections = tuple(
        (keys[match["header"]], text[match.end() : end])
        for match, end in zip(matches, ends, strict=True)
    )
    return text[: matches[0].start()] if matches else text, sections


class NarrativeMeasure(NamedTuple):
    """Změřený narativ zprávy: počet slov a počet narativních značek bez věkové značky."""

    words: int
    marks: int


def measure_narrative(
    sections: Sections, plan: NotePlan, structure: StructureModel
) -> NarrativeMeasure:
    """Změří narativ: těla sekcí s délkou v plánu bez podnadpisů.

    Returns:
        Počet slov narativu a počet narativních značek bez věkové značky.
    """
    subheadings = line_headers(
        subheading.header for _, subheading in structure.subheadings
    )
    bodies = [
        subheadings.sub("", body) for key, body in sections if key in plan.section_words
    ]
    words = sum(len(body.split()) for body in bodies)
    marks = sum(deid_markers(body) - len(AGE_MARKER.findall(body)) for body in bodies)
    return NarrativeMeasure(words, marks)
