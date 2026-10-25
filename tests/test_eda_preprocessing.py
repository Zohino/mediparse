"""Vstupní bod EDA, který měří předzpracování TF-IDF a zkratky diagnóz pro sešit 9."""

from __future__ import annotations

import pytest

from mediparse.entrypoints.eda_preprocessing import (
    PREPROCESSED,
    RAW,
    diagnosis_acronyms,
    note_preprocessing,
    note_vocabulary,
)
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

ACRONYMS = ["AF", "CKD", "COPD", "DM", "DM2", "ESRD"]
BASE_COLUMNS = {"note_id", "tokens_raw", "tokens_preprocessed"}


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


def test_empty_input_keeps_every_column() -> None:
    """Prázdná dávka dá všechny sloupce, aby R složilo typovaný data.frame."""
    columns = note_preprocessing(REPOSITORY_DIAGNOSIS_CANDIDATES, [], [])
    vocabulary = note_vocabulary(REPOSITORY_DIAGNOSIS_CANDIDATES, [])

    assert set(columns) == BASE_COLUMNS | set(ACRONYMS)
    assert all(column == [] for column in columns.values())
    assert vocabulary == {"variant": [], "token": [], "notes": []}


def test_different_lengths_raise_value_error() -> None:
    """Zprávy a identifikátory musí být stejně dlouhé."""
    with pytest.raises(ValueError, match=r"shorter|argument"):
        note_preprocessing(REPOSITORY_DIAGNOSIS_CANDIDATES, ["n1", "n2"], ["text"])


def test_vocabulary_lists_distinct_tokens_by_variant_with_note_counts() -> None:
    """Slovník dávky: distinct tokeny po variantách s počtem zpráv, které je mají."""
    vocabulary = note_vocabulary(REPOSITORY_DIAGNOSIS_CANDIDATES, ["Pt AF pt", "Pt DM"])

    rows = list(
        zip(
            vocabulary["variant"], vocabulary["token"], vocabulary["notes"], strict=True
        )
    )
    assert rows == [
        (RAW, "af", 1),
        (RAW, "dm", 1),
        (RAW, "pt", 2),
        (PREPROCESSED, "atrial", 1),
        (PREPROCESSED, "diabetes", 1),
        (PREPROCESSED, "fibrillation", 1),
        (PREPROCESSED, "pt", 2),
    ]


def test_every_acronym_expands_in_the_model_analyzer() -> None:
    """Počet zkratek v mostu odpovídá rozvinutí v modelu: kanonický tvar je v tokenech."""
    normalization = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES)
    analyzer = tfidf_vectorizer(normalization).build_analyzer()

    for acronym, form in normalization.acronyms.items():
        text = f"Pt has {acronym} today"
        counted = note_preprocessing(REPOSITORY_DIAGNOSIS_CANDIDATES, ["n"], [text])

        assert counted[acronym] == [1]
        assert form.split()[0] in analyzer(text)
