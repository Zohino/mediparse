"""Výpočet tokenů, slovníků a zkratek zpráv přes use case."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from tests.note_features_support import MemoryFeatures, compute
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def diagnosis_acronyms(candidates: Path) -> list[str]:
    """Vrátí seřazené klíče konfigurace pro původní očekávání.

    Returns:
        Zkratky konfigurace.
    """
    return sorted(load_text_normalization(candidates).acronyms)


def note_preprocessing(
    candidates: Path, note_ids: Sequence[str], texts: Sequence[str]
) -> dict[str, list]:
    """Převede výsledek use casu na původní sloupcová očekávání.

    Returns:
        Počty tokenů a zkratek.
    """
    sink = MemoryFeatures()
    compute(list(zip(note_ids, texts, strict=True)), sink)
    return {
        "note_id": [row.note_id for row in sink.preprocessing],
        "tokens_raw": [row.tokens_raw for row in sink.preprocessing],
        "tokens_preprocessed": [row.tokens_preprocessed for row in sink.preprocessing],
        **{
            key: [row.occurrences for row in sink.acronyms if row.acronym == key]
            for key in diagnosis_acronyms(candidates)
        },
    }


ACRONYMS = ["AF", "CKD", "COPD", "DM", "DM2", "ESRD"]
TWICE: Final = 2


def test_diagnosis_acronyms_are_sorted_keys_of_the_config() -> None:
    """Sešit dostane zkratky z configu, ne vlastní seznam."""
    assert diagnosis_acronyms(REPOSITORY_DIAGNOSIS_CANDIDATES) == ACRONYMS


def test_tokens_before_and_after_with_acronym_count() -> None:
    """Tokeny před předzpracováním jsou výchozí tokenizace, po něm tokeny modelu."""
    columns = note_preprocessing(
        REPOSITORY_DIAGNOSIS_CANDIDATES, ["n1"], ["Pt 5mg ___ AF"]
    )

    assert columns["note_id"] == ["n1"]
    assert columns["tokens_raw"] == [4]
    assert columns["tokens_preprocessed"] == [6]
    assert columns["AF"] == [1]
    assert {key: value for key, value in columns.items() if key in ACRONYMS} == {
        "AF": [1],
        "CKD": [0],
        "COPD": [0],
        "DM": [0],
        "DM2": [0],
        "ESRD": [0],
    }


def test_acronyms_are_counted_after_normalization() -> None:
    """Zkratka přilepená ke značce ___ se najde, protože normalizace jde první."""
    columns = note_preprocessing(
        REPOSITORY_DIAGNOSIS_CANDIDATES, ["n1"], ["___COPD and DM2, DM; AFib af"]
    )

    assert [columns[acronym] for acronym in ACRONYMS] == [[0], [0], [1], [1], [1], [0]]


def test_empty_input_has_no_features() -> None:
    """Prázdná dávka nezapíše sekce, tokeny, slovníky ani zkratky."""
    sink = MemoryFeatures()
    assert compute([], sink) == 0
    assert sink.sections == []
    assert sink.preprocessing == []
    assert sink.acronyms == []


def test_different_lengths_raise_value_error() -> None:
    """Zprávy a identifikátory musí být stejně dlouhé."""
    with pytest.raises(ValueError, match="argument 2 is shorter"):
        note_preprocessing(REPOSITORY_DIAGNOSIS_CANDIDATES, ["n1", "n2"], ["text"])


def test_every_acronym_expands_in_the_model_analyzer() -> None:
    """Počet zkratek v mostu odpovídá rozvinutí v modelu: kanonický tvar je v tokenech."""
    normalization = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES)
    analyzer = tfidf_vectorizer(normalization).build_analyzer()

    for acronym, form in normalization.acronyms.items():
        text = f"Pt has {acronym} today"
        counted = note_preprocessing(REPOSITORY_DIAGNOSIS_CANDIDATES, ["n"], [text])

        assert counted[acronym] == [1]
        assert form.split()[0] in analyzer(text)


def test_acronym_order_with_marker_matches_model_expansion() -> None:
    """Zkratka přilepená ke značce se počítá dvakrát a model ji dvakrát rozvine."""
    normalization = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES)
    analyzer = tfidf_vectorizer(normalization).build_analyzer()

    for acronym, form in normalization.acronyms.items():
        text = f"___{acronym} and {acronym}."
        counted = note_preprocessing(REPOSITORY_DIAGNOSIS_CANDIDATES, ["n"], [text])

        assert counted[acronym] == [TWICE]
        assert analyzer(text).count(form.split()[0]) == TWICE


def test_vocabulary_is_sorted_distinct_per_note() -> None:
    """Slovník každé zprávy je seřazený a distinct, zkratky mají i nulové řádky."""
    sink = MemoryFeatures()
    assert compute([("n1", "Pt AF pt"), ("n2", "Pt DM")], sink) == TWICE
    assert [row.vocabulary_raw for row in sink.preprocessing] == [
        ("af", "pt"),
        ("dm", "pt"),
    ]
    assert [row.vocabulary_preprocessed for row in sink.preprocessing] == [
        ("atrial", "fibrillation", "pt"),
        ("diabetes", "pt"),
    ]
    assert [(row.note_id, row.acronym, row.occurrences) for row in sink.acronyms] == [
        ("n1", "AF", 1),
        ("n1", "CKD", 0),
        ("n1", "COPD", 0),
        ("n1", "DM", 0),
        ("n1", "DM2", 0),
        ("n1", "ESRD", 0),
        ("n2", "AF", 0),
        ("n2", "CKD", 0),
        ("n2", "COPD", 0),
        ("n2", "DM", 1),
        ("n2", "DM2", 0),
        ("n2", "ESRD", 0),
    ]
