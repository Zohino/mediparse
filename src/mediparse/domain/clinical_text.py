"""Normalizace, rozvinutí zkratek a přepis variant a překlepů pro TF-IDF."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from mediparse.domain.language import Language
from mediparse.domain.note_text import STANDALONE_DEID, alternation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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


def find_diagnosis_acronyms(text: str, acronyms: Mapping[str, str]) -> list[str]:
    """Najde zkratky diagnóz týmž vzorem, jaký rozvinutí nahrazuje.

    Má běžet nad ``normalize_clinical_text``, ze stejného důvodu jako rozvinutí.

    Returns:
        Zkratky v pořadí výskytu; prázdný slovník dá prázdný seznam.
    """
    if not acronyms:
        return []
    return [match[0] for match in _acronym_pattern(acronyms).finditer(text)]


def _variant_key(text: str) -> str:
    return " ".join(text.lower().split())


def _variant_keys(variants: Iterable[str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for variant in variants:
        key = _variant_key(variant)
        if key != variant.lower():
            message = f"Varianta nemá sloučené bílé znaky: {variant!r}."
            raise ValueError(message)
        if key in canonical:
            message = f"Duplicitní varianta: {variant}."
            raise ValueError(message)
        canonical[key] = variant
    return canonical


def _variant_pattern(variants: Iterable[str]) -> re.Pattern[str]:
    # Víceslovná varianta přijímá bílé znaky stejně jako zmínka v sešitu 8.
    alternatives = alternation(variants).replace(r"\ ", r"\s+")
    # ASCII brání záměně dlouhého s (U+017F) a znaku Kelvina za písmena slovníku.
    return re.compile(
        f"{_WORD_START}(?:{alternatives}){_WORD_END}",
        re.IGNORECASE | re.ASCII,
    )


def find_diagnosis_variants(text: str, variants: Iterable[str]) -> list[str]:
    """Najde celé varianty bez rozlišení velikosti písmen po normalizaci textu.

    Returns:
        Tvary z konfigurace v pořadí výskytu; prázdný slovník dá prázdný seznam.

    Raises:
        ValueError: Tvary jsou duplicitní nebo nemají sloučené bílé znaky.
    """
    canonical = _variant_keys(variants)
    if not canonical:
        return []
    return [
        canonical[_variant_key(match[0])]
        for match in _variant_pattern(canonical).finditer(text)
    ]


def rewrite_diagnosis_variants(text: str, variants: Mapping[str, str]) -> str:
    """Přepíše celé varianty a překlepy bez rozlišení velikosti písmen.

    Returns:
        Text s náhradami ze slovníku; prázdný slovník nechá text beze změny.
    """
    if not variants:
        return text
    canonical = _variant_keys(variants)
    pattern = _variant_pattern(canonical)
    # Oprava překlepu může odhalit víceslovnou variantu až po první náhradě.
    # TextNormalization omezuje řetězy návazností a vyžaduje slovní okraje variant,
    # proto dva průchody stačí i při změně sousedního znaku náhradou.
    for _ in range(2):
        text = pattern.sub(
            lambda match: variants[canonical[_variant_key(match[0])]], text
        )
    return text


def _form_words(form: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", form.lower()))


def _replacement_can_form_variant(
    replacement_words: tuple[str, ...], variant_words: tuple[str, ...]
) -> bool:
    if any(
        variant_words[start : start + len(replacement_words)] == replacement_words
        for start in range(len(variant_words) - len(replacement_words) + 1)
    ):
        return True
    return any(
        replacement_words[-overlap:] == variant_words[:overlap]
        or replacement_words[:overlap] == variant_words[-overlap:]
        for overlap in range(1, min(len(replacement_words) - 1, len(variant_words)) + 1)
    )


def _composed_replacement_chains(
    variants: Mapping[str, str],
) -> list[tuple[str, str]]:
    variant_words = {variant: _form_words(variant) for variant in variants}
    next_variants = {}
    for variant, replacement in variants.items():
        replacement_words = _form_words(replacement)
        next_variants[variant] = {
            target
            for target, words in variant_words.items()
            if _replacement_can_form_variant(replacement_words, words)
        }
    composable = {target for targets in next_variants.values() for target in targets}
    return sorted(
        (variant, target) for variant in composable for target in next_variants[variant]
    )


class TextNormalization(BaseModel):
    """Záznam normalizace vstupu TF-IDF: slovníky zkratek, variant a překlepů.

    Tentýž objekt dostane trenér i run manifest. Klíče se seřadí, aby byl JSON
    deterministický. Náhrady nesmějí obsahovat tvary ani zkratky slovníků jako celá slova,
    jinak by přepis nebyl idempotentní. Náhrada se s okolím smí složit do další varianty,
    jejíž náhrada už další variantu složit nesmí. Varianta musí začínat i končit
    písmenem a-z, číslicí nebo podtržítkem po převodu na malá písmena; společně
    s pravidlem návaznosti to zajišťuje, že dva průchody přepisu stačí.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    acronyms: dict[
        Annotated[str, StringConstraints(min_length=1)],
        Annotated[str, StringConstraints(min_length=1)],
    ]

    variants: dict[
        Annotated[str, StringConstraints(min_length=1)],
        Annotated[str, StringConstraints(min_length=1)],
    ]

    @field_validator("acronyms", "variants")
    @classmethod
    def _sorted_dictionary(cls, dictionary: dict[str, str]) -> dict[str, str]:
        return dict(sorted(dictionary.items()))

    @model_validator(mode="after")
    def _dictionaries_are_idempotent(self) -> TextNormalization:
        _variant_keys(self.variants)
        invalid_edges = [
            variant
            for variant in self.variants
            if re.fullmatch(r"[a-z0-9_](?:.*[a-z0-9_])?", variant.lower(), re.DOTALL)
            is None
        ]
        if invalid_edges:
            message = f"Varianta má neslovní okrajový znak: {invalid_edges}."
            raise ValueError(message)
        collisions = [
            variant
            for variant in self.variants
            if find_diagnosis_variants(variant, self.acronyms)
        ]
        if collisions:
            message = f"Varianty obsahují zkratky slovníku: {sorted(collisions)}."
            raise ValueError(message)
        for form in (*self.acronyms.values(), *self.variants.values()):
            if find_diagnosis_variants(form, self.variants) or (
                self.acronyms and _acronym_pattern(self.acronyms).search(form)
            ):
                message = f"Náhrada obsahuje variantu nebo zkratku slovníku: {form}"
                raise ValueError(message)
        chains = _composed_replacement_chains(self.variants)
        if chains:
            message = (
                f"Řetěz složených náhrad variant má více než jednu návaznost: {chains}."
            )
            raise ValueError(message)
        return self
