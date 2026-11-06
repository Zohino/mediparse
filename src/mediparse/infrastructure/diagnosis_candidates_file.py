"""Načtení kódů ICD, zkratek a variant názvů pětice ze sdíleného configu kandidátů."""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mediparse.domain.clinical_text import TextNormalization, find_diagnosis_variants
from mediparse.domain.diagnosis import Diagnosis, PerDiagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from collections.abc import Mapping
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
    if not variants:
        vocabulary = _vocabulary(text)
        return TextNormalization(acronyms=_acronyms(vocabulary.names), variants={})
    candidates = _Candidates.model_validate_json(text).candidates
    names = _selected(candidates, _MisspellingNames)
    return _normalization_with_variants(text, names)


def _acronyms(names: Mapping[Diagnosis, _Names]) -> dict[str, str]:
    return {
        acronym: entry.keywords_en[0]
        for entry in names.values()
        for acronym in entry.acronyms_en
    }


def _normalization_with_variants(
    text: str,
    names: Mapping[Diagnosis, _VariantNames],
) -> TextNormalization:
    pairs = [
        (variant, entry.keywords_en[0])
        for entry in names.values()
        for variant in entry.variants_en
    ]
    pairs.extend(
        (form, word)
        for entry in names.values()
        if isinstance(entry, _MisspellingNames)
        for form, word in entry.misspellings_en.items()
    )
    # Kontrola před dict zachová i přesné duplicity napříč diagnózami a poli.
    try:
        find_diagnosis_variants("", [form for form, _ in pairs])
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    return TextNormalization(
        acronyms=_acronyms(_vocabulary(text).names), variants=dict(pairs)
    )


def _vocabulary(text: str) -> _Vocabulary:
    candidates = _Candidates.model_validate_json(text).candidates
    return _Vocabulary(names=_selected(candidates, _Names))


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
    candidates = _Candidates.model_validate_json(text).candidates
    parsed = _selected(candidates, _VariantNames)
    _normalization_with_variants(text, parsed)
    return dict(
        sorted(
            (variant, diagnosis)
            for diagnosis, names in parsed.items()
            for variant in names.variants_en
        )
    )


def _selected[T: BaseModel](
    candidates: Mapping[str, dict[str, Any]], model: type[T]
) -> dict[Diagnosis, T]:
    return {
        diagnosis: model.model_validate(candidates[diagnosis.value])
        for diagnosis in Diagnosis
        if diagnosis.value in candidates
    }
