"""Text zprávy: de-identifikační značka a její počet, věková značka."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

DEID: Final = "___"
# Delší řady podtržítek nejsou značka (hlásí je kontrola cizí formy značky). Vzor
# proto nelze zjednodušit na str.count, ten by ____ započítal jako značku.
STANDALONE_DEID: Final = rf"(?<!_){re.escape(DEID)}(?!_)"
_DEID_PATTERN: Final = re.compile(STANDALONE_DEID)
# Lookahead místo \b přijme i y.o. končící tečkou; jedno m/f patří k masce
# věku s přilepeným pohlavím. Protějšek pro RE2 je v eda/R/text.R.
AGE_SUFFIX: Final = r"[ -]?(?:years?[ -]?old|yo|y/o|y\.o\.)[mf]?(?![a-z])"
AGE_MARKER: Final = re.compile(rf"{STANDALONE_DEID}{AGE_SUFFIX}", re.IGNORECASE)


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
