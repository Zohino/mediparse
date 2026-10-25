"""Normalizace klinického textu pro TF-IDF: zástupné tokeny a rozvinutí zkratek."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from mediparse.domain.language import Language
from mediparse.domain.note_text import STANDALONE_DEID, alternation

if TYPE_CHECKING:
    from collections.abc import Mapping

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
# slova (DM2, HbA1c) se nemění, staví na něm rozvinutí zkratek diagnóz. Druhý
# lookbehind chrání desetinnou část čísla přilepeného k písmenu (K3.5): bez něj by
# se nahradila za tečkou. Řadová přípona se pohltí jen tehdy, když za ní nepokračuje
# slovo ani číslice, jinak by 1st2nd nebylo idempotentní.
_NUMBER: Final = re.compile(
    r"(?<![A-Za-z0-9_])(?<![0-9][.,])[0-9]+(?:[.,][0-9]+)*"
    r"(?:(?i:st|nd|rd|th)(?![A-Za-z0-9_]))?"
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


# Hranice jako ASCII \b v RE2 (sešit 8) a jako lookaheady výše: Python \b je Unicode
# a AFé by se tam choval jinak než zmínky v EDA. Zkratky se rozlišují podle velikosti
# písmen, protože mi, af a hf malými písmeny jsou běžná slova.
_WORD_START: Final = "(?<![A-Za-z0-9_])"
_WORD_END: Final = "(?![A-Za-z0-9_])"


def _acronym_pattern(acronyms: Mapping[str, str]) -> re.Pattern[str]:
    return re.compile(f"{_WORD_START}(?:{alternation(acronyms)}){_WORD_END}")


def expand_diagnosis_acronyms(text: str, acronyms: Mapping[str, str]) -> str:
    """Nahradí zkratky diagnóz jejich kanonickým tvarem.

    Zkratka se rozvine jen jako celé slovo a s rozlišením velikosti písmen. Má běžet
    po ``normalize_clinical_text``, protože ``_`` je znak slova a ``___COPD`` by
    se jinak nerozvinul. Delší zkratka vyhrává nad svou předponou (``DM2`` nad ``DM``).

    Returns:
        Text s rozvinutými zkratkami; prázdný slovník nechá text beze změny.
    """
    if not acronyms:
        return text
    # Vzor se kompiluje při každém volání, protože model nese slovník jako data, ne
    # vzor; kompilaci zrychluje cache modulu re.
    return _acronym_pattern(acronyms).sub(lambda match: acronyms[match[0]], text)


class TextNormalization(BaseModel):
    """Záznam normalizace vstupu TF-IDF: slovník zkratek diagnóz a jejich tvarů.

    Tentýž objekt dostane trenér i run manifest. Klíče se seřadí, aby byl JSON
    deterministický. Tvar nesmí obsahovat zkratku slovníku jako celé slovo, jinak by
    rozvinutí nebylo idempotentní.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    acronyms: dict[
        Annotated[str, StringConstraints(min_length=1)],
        Annotated[str, StringConstraints(min_length=1)],
    ]

    @field_validator("acronyms")
    @classmethod
    def _sorted_and_idempotent(cls, acronyms: dict[str, str]) -> dict[str, str]:
        pattern = _acronym_pattern(acronyms)
        for acronym, form in acronyms.items():
            if pattern.search(form):
                message = f"Tvar zkratky {acronym} obsahuje zkratku slovníku: {form}"
                raise ValueError(message)
        return dict(sorted(acronyms.items()))
