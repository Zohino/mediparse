"""Text syntetické zprávy: de-identifikační značka, sekce podle hlaviček na začátku řádku a věková značka."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

DEID: Final = "___"
AGE_SUFFIX: Final = r"[ -]?(?:years?[ -]old|y/?o)\b"
AGE_MARKER: Final = re.compile(rf"{re.escape(DEID)}{AGE_SUFFIX}", re.IGNORECASE)

type Sections = tuple[tuple[str, str], ...]


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
