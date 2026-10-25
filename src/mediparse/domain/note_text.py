"""Text zprávy: de-identifikační značka a její počet, sekce podle hlaviček na začátku řádku, věková značka."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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
        Preambuli a dvojice klíč sekce, tělo sekce v pořadí textu; text bez
        hlavičky je celý preambulí a sekce nemá.
    """
    keys = {header: key for key, header in headers.items()}
    matches = list(line_headers(keys).finditer(text))
    starts = [match.start() for match in matches] + [len(text)]
    sections = tuple(
        (keys[match["header"]], text[match.end() : end])
        for match, end in zip(matches, starts[1:], strict=True)
    )
    return text[: starts[0]], sections
