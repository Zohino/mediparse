"""Lexikon sekcí, normalizace hlavičky a rozpoznání hlaviček v textu zprávy."""

from __future__ import annotations

import json
from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.section_headers import SectionLexicon, normalized


def _lexicon_json(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "sections": {
            "chief_complaint": {
                "label_cs": "Hlavní obtíže",
                "synonyms": ["Chief Complaint"],
            },
            "allergies": {"label_cs": "Alergie", "synonyms": ["Allergies"]},
        },
        "preamble": ["Name", "Sex"],
    }
    return document | overrides


def _lexicon(**overrides: object) -> SectionLexicon:
    return SectionLexicon.model_validate_json(json.dumps(_lexicon_json(**overrides)))


@pytest.mark.parametrize(
    "header", ["CHIEF  COMPLAINT ", "Chief Complaint.", "chief\tcomplaint"]
)
def test_normalization_merges_spelling_variants(header: str) -> None:
    """Velikost písmen, běh bílých znaků a koncové mezery a tečky se sloučí."""
    assert normalized(header) == "chief complaint"


def test_minimal_lexicon_is_accepted() -> None:
    """Lexikon se sekcemi a preambulí projde schématem."""
    assert set(_lexicon().sections) == {"chief_complaint", "allergies"}


def _section(*synonyms: str, label_cs: str = "A") -> dict[str, object]:
    return {"label_cs": label_cs, "synonyms": list(synonyms)}


INVALID_LEXICONS: Final = {
    "sekce bez synonym": {"sections": {"a": _section()}},
    "prázdné label_cs": {"sections": {"a": _section("Allergies", label_cs="")}},
    "kolize dvou sekcí po normalizaci": {
        "sections": {"a": _section("Allergies"), "b": _section("ALLERGIES.")}
    },
    "kolize sekce s preamble": {"preamble": ["Name", "allergies"]},
    "synonymum s číslicí": {"sections": {"a": _section("Type 2 Diabetes")}},
    "malé první písmeno": {"sections": {"a": _section("allergies")}},
    "normalizovaný tvar pod 3 znaky": {"sections": {"a": _section("Ab")}},
    "neznámý klíč na nejvyšší úrovni": {"extra": 1},
}


@pytest.mark.parametrize(
    "overrides", INVALID_LEXICONS.values(), ids=INVALID_LEXICONS.keys()
)
def test_invalid_lexicon_is_rejected(overrides: dict[str, object]) -> None:
    """Každé porušení pravidla lexikonu zastaví validaci."""
    with pytest.raises(ValidationError):
        _lexicon(**overrides)
