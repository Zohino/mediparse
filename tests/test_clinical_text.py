"""Normalizace klinického textu: věk, značka a čísla s jednotkami na zástupné tokeny."""

from __future__ import annotations

import re
import string
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.feature_extraction.text import TfidfVectorizer

from mediparse.domain.clinical_text import (
    AGE_PLACEHOLDER,
    DEID_PLACEHOLDER,
    NORMALIZATION_LANGUAGE,
    NUMBER_PLACEHOLDER,
    normalize_clinical_text,
)
from mediparse.domain.language import Language
from mediparse.domain.note_text import deid_markers

_ANALYZER: Final = TfidfVectorizer().build_analyzer()
_UNREPLACED_NUMBER: Final = re.compile(r"(?<![A-Za-z0-9_])[0-9]")
_TEXT: Final = st.text(
    alphabet=string.ascii_letters + string.digits + "_ -.,/%\n" + "é٣", max_size=40
)


def _tokens(text: str) -> list[str]:
    return _ANALYZER(normalize_clinical_text(text))


def test_normalization_is_english_only() -> None:
    """Pravidla jsou anglická, jazyk normalizace je angličtina."""
    assert NORMALIZATION_LANGUAGE is Language.EN


@pytest.mark.parametrize(
    "form",
    [
        "___ year old",
        "___ years old",
        "___-year-old",
        "___ yearsold",
        "___yo",
        "___ y/o",
        "___ y.o.",
        "___ YO",
    ],
)
def test_every_age_form_from_eda_becomes_age_placeholder(form: str) -> None:
    """Všechny tvary věku naměřené v MIMIC dají týž zástupný token."""
    assert normalize_clinical_text(f"a {form} woman").split() == [
        "a",
        AGE_PLACEHOLDER,
        "woman",
    ]


@pytest.mark.parametrize("text", ["___ yogurt", "___ y/om"])
def test_age_suffix_continuing_into_letters_is_not_age(text: str) -> None:
    """Přípona věku, za níž pokračují písmena, je jiné slovo."""
    assert AGE_PLACEHOLDER not in normalize_clinical_text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr.___", f"Mr. {DEID_PLACEHOLDER} "),
        ("___'s", f" {DEID_PLACEHOLDER} 's"),
        ("at ___pm", f"at  {DEID_PLACEHOLDER} pm"),
        ("a ___, b", f"a  {DEID_PLACEHOLDER} , b"),
    ],
)
def test_other_marker_becomes_deid_placeholder_with_spaces(
    text: str, expected: str
) -> None:
    """Ostatní značka dostane zástupný token oddělený mezerou od přilepeného okolí."""
    assert normalize_clinical_text(text) == expected


@pytest.mark.parametrize("text", ["____", "____ year old", "____yo", "_2", "__"])
def test_longer_underscore_runs_and_underscore_numbers_stay(text: str) -> None:
    """Delší řady podtržítek nejsou značka a číslo za podtržítkem není samostatné."""
    assert normalize_clinical_text(text) == text


def test_dose_spelling_does_not_change_tokens() -> None:
    """Dávka bez mezery a s mezerou dá po normalizaci stejné tokeny."""
    assert _tokens("metformin 500mg") == _tokens("metformin 500 mg")
    assert _tokens("metformin 500mg") == ["metformin", NUMBER_PLACEHOLDER, "mg"]


@pytest.mark.parametrize(
    ("text", "numbers"),
    [("1.8", 1), ("98.6", 1), ("1,000", 1), ("126/74", 2), ("2nd", 1), ("5.4%", 1)],
)
def test_numbers_become_one_placeholder_each(text: str, numbers: int) -> None:
    """Desetinné číslo a číslo s tisícovým oddělovačem je jedno, řadová přípona se pohltí."""
    assert normalize_clinical_text(text).split().count(NUMBER_PLACEHOLDER) == numbers
    assert not _UNREPLACED_NUMBER.search(normalize_clinical_text(text))


def test_percent_is_separated_from_number() -> None:
    """Procento přilepené k číslu se oddělí mezerou."""
    assert normalize_clinical_text("5.4%").split() == [NUMBER_PLACEHOLDER, "%"]


@pytest.mark.parametrize("text", ["DM2", "HbA1c", "B12", "T2DM", "q6h", "x2"])
def test_number_inside_word_stays(text: str) -> None:
    """Číslo uvnitř slova a zkratky s číslem se nemění, včetně velikosti písmen."""
    assert normalize_clinical_text(text) == text


def test_empty_text_stays_empty() -> None:
    """Prázdný text zůstane prázdný."""
    assert not normalize_clinical_text("")


@given(_TEXT)
def test_normalization_is_idempotent(text: str) -> None:
    """Druhý průchod už nic nemění."""
    once = normalize_clinical_text(text)
    assert normalize_clinical_text(once) == once


@given(_TEXT)
def test_no_standalone_marker_remains(text: str) -> None:
    """Po normalizaci nezbude samostatná značka ___."""
    assert deid_markers(normalize_clinical_text(text)) == 0


@given(_TEXT)
def test_no_replaceable_number_remains(text: str) -> None:
    """Každá řada ASCII číslic stojí hned za písmenem, číslicí nebo podtržítkem."""
    assert not _UNREPLACED_NUMBER.search(normalize_clinical_text(text))


@given(st.text(alphabet=string.ascii_letters + " .,\n-", max_size=40))
def test_text_without_underscore_and_digits_stays(text: str) -> None:
    """Text bez podtržítka a číslic se nezmění."""
    assert normalize_clinical_text(text) == text
