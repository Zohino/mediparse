"""Načtení kódů ICD, zkratek a variant názvů pětice ze sdíleného configu kandidátů."""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis, PerDiagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class _Prefixes(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    icd9_prefixes: tuple[str, ...]
    icd10_prefixes: tuple[str, ...]


class _Candidates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    candidates: dict[str, dict[str, Any]]


def load_diagnosis_codes(path: Path) -> DiagnosisCodes:
    """Přečte předpony diagnóz pětice; ostatní kandidáty a pole ignoruje.

    Returns:
        Předpony všech diagnóz, které prošly doménovými invarianty.
    """
    return parse_file(path, _parse)


def _parse(text: str) -> DiagnosisCodes:
    candidates = _Candidates.model_validate_json(text).candidates
    parsed = _selected(candidates, _Prefixes)
    return DiagnosisCodes(
        prefixes={
            diagnosis: {
                IcdVersion.NINE: prefixes.icd9_prefixes,
                IcdVersion.TEN: prefixes.icd10_prefixes,
            }
            for diagnosis, prefixes in parsed.items()
        }
    )


class _Names(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    keywords_en: tuple[str, ...]
    acronyms_en: tuple[str, ...]

    @model_validator(mode="after")
    def _acronyms_have_canonical_form(self) -> _Names:
        if self.acronyms_en and not self.keywords_en:
            msg = f"Zkratky {self.acronyms_en} nemají keywords_en s kanonickým tvarem."
            raise ValueError(msg)
        return self


class _Vocabulary(BaseModel):
    model_config = ConfigDict(frozen=True)

    names: PerDiagnosis[_Names]

    @model_validator(mode="after")
    def _acronyms_are_unique(self) -> _Vocabulary:
        seen: set[str] = set()
        for names in self.names.values():
            duplicates = seen.intersection(names.acronyms_en)
            if duplicates:
                msg = f"Zkratky u více diagnóz: {sorted(duplicates)}."
                raise ValueError(msg)
            seen.update(names.acronyms_en)
        return self


def load_text_normalization(path: Path, *, variants: bool) -> TextNormalization:
    """Přečte zkratky pětice a volitelně varianty a překlepy.

    Kanonický tvar je první položka ``keywords_en`` diagnózy. Zkratky ostatních
    kandidátů se nenačítají, protože klasifikátor zná jen pětici.

    Returns:
        Slovníky zkratek a zapnutých variant a překlepů pro TF-IDF.
    """
    return parse_file(path, partial(_parse_normalization, variants=variants))


def _parse_normalization(text: str, *, variants: bool) -> TextNormalization:
    vocabulary = _vocabulary(text)
    pairs: list[tuple[str, str]] = []
    if variants:
        names = _selected(vocabulary, _MisspellingNames)
        pairs = [*_variant_pairs(names), *_misspelling_pairs(names)]
    return TextNormalization(
        acronyms=_acronyms(vocabulary), variants=_unique_forms(pairs)
    )


def _acronyms(candidates: Mapping[str, dict[str, Any]]) -> dict[str, str]:
    vocabulary = _Vocabulary(names=_selected(candidates, _Names))
    return {
        acronym: entry.keywords_en[0]
        for entry in vocabulary.names.values()
        for acronym in entry.acronyms_en
    }


def _variant_pairs(names: Mapping[Diagnosis, _VariantNames]) -> list[tuple[str, str]]:
    return [
        (variant, entry.keywords_en[0])
        for entry in names.values()
        for variant in entry.variants_en
    ]


def _misspelling_pairs(
    names: Mapping[Diagnosis, _MisspellingNames],
) -> list[tuple[str, str]]:
    return [
        (form, word)
        for entry in names.values()
        for form, word in entry.misspellings_en.items()
    ]


def _unique_forms(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    forms: dict[str, str] = {}
    for form, word in pairs:
        if form in forms:
            msg = f"Duplicitní varianta: {form}."
            raise InvalidInputError(msg)
        forms[form] = word
    return forms


def _vocabulary(text: str) -> dict[str, dict[str, Any]]:
    return _Candidates.model_validate_json(text).candidates


class _VariantNames(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")

    keywords_en: Annotated[list[str], Field(min_length=1)]
    acronyms_en: list[str]
    variants_en: Annotated[
        list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]],
        Field(min_length=1),
    ]


class _MisspellingNames(_VariantNames):
    misspellings_en: dict[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)],
        Annotated[str, StringConstraints(min_length=1)],
    ]

    @model_validator(mode="after")
    def _targets_are_keyword_words(self) -> _MisspellingNames:
        words = set(re.findall(r"[A-Za-z]+", " ".join(self.keywords_en)))
        for word in self.misspellings_en.values():
            if word not in words:
                msg = f"Cíl překlepu {word} není slovo keywords_en téže diagnózy."
                raise ValueError(msg)
        return self


def load_diagnosis_variants(path: Path) -> dict[str, Diagnosis]:
    """Přečte varianty pětice, ostatní kandidáty a pole sdíleného configu ignoruje.

    Returns:
        Seřazený slovník varianta → diagnóza bez duplicit a kolizí se zkratkami.
    """
    return parse_file(path, _parse_variants)


def _parse_variants(text: str) -> dict[str, Diagnosis]:
    vocabulary = _vocabulary(text)
    names = _selected(vocabulary, _VariantNames)
    normalization = TextNormalization(
        acronyms=_acronyms(vocabulary),
        variants=_unique_forms(_variant_pairs(names)),
    )
    diagnoses = {
        variant: diagnosis
        for diagnosis, entry in names.items()
        for variant in entry.variants_en
    }
    return {variant: diagnoses[variant] for variant in normalization.variants}


def _selected[T: BaseModel](
    candidates: Mapping[str, dict[str, Any]], model: type[T]
) -> dict[Diagnosis, T]:
    return {
        diagnosis: model.model_validate(candidates[diagnosis.value])
        for diagnosis in Diagnosis
        if diagnosis.value in candidates
    }
