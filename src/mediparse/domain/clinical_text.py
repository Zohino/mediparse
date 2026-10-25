"""Normalizace klinického textu pro TF-IDF: věk, značka a čísla na zástupné tokeny."""

from __future__ import annotations

import re
from typing import Final

from mediparse.domain.language import Language
from mediparse.domain.note_text import STANDALONE_DEID

NORMALIZATION_LANGUAGE: Final = Language.EN
AGE_PLACEHOLDER: Final = "__age__"
DEID_PLACEHOLDER: Final = "__deid__"
NUMBER_PLACEHOLDER: Final = "__num__"

# Výčet přípon převzatý z AGE_MASK v eda/R/text.R, aby sešit 9 měřil nad MIMIC
# tytéž tvary. Vzor se sestavuje ze STANDALONE_DEID: holé ___ by z ____ year old
# udělalo "_ __age__". Místo \b stojí lookahead, protože \b za tečkou v y.o.
# nic nenajde.
_AGE: Final = re.compile(
    rf"{STANDALONE_DEID}[ -]?(?:years?[ -]?old|yo|y/o|y\.o\.)(?![A-Za-z])",
    re.IGNORECASE,
)
_DEID: Final = re.compile(STANDALONE_DEID)
# [0-9] místo \d, které v Pythonu zachytí i číslice jiných písem. Číslo uvnitř
# slova (DM2, HbA1c) se nemění, staví na něm rozvinutí zkratek diagnóz.
_NUMBER: Final = re.compile(
    r"(?<![A-Za-z0-9_])[0-9]+(?:[.,][0-9]+)*(?:(?:st|nd|rd|th)(?![A-Za-z]))?"
)


def normalize_clinical_text(text: str) -> str:
    """Sjednotí tvary věku, značky ``___`` a čísla s jednotkami na zástupné tokeny.

    Zástupný token má mezery kolem sebe, takže ``500mg`` a ``___pm`` nezůstanou
    slepené s okolím. Velikost písmen se zachová.

    Returns:
        Text, v němž věk, ostatní samostatné značky a čísla nahradily zástupné tokeny.
    """
    # Věk musí jít před obecnou značkou, jinak z ___ year old vznikne __deid__ year old.
    text = _AGE.sub(f" {AGE_PLACEHOLDER} ", text)
    text = _DEID.sub(f" {DEID_PLACEHOLDER} ", text)
    return _NUMBER.sub(f" {NUMBER_PLACEHOLDER} ", text)
