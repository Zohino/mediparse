"""Vstupní bod EDA, který rozdělí zprávy parserem sekcí na sloupce pro sešit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain import section_headers
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints import eda_sections
from mediparse.entrypoints.eda_sections import OTHER, note_sections
from tests.support import REPOSITORY_SECTION_HEADERS

if TYPE_CHECKING:
    from pathlib import Path

COLUMNS = {"note_id", "position", "section", "span_chars", "body_chars"}
WITH_PREAMBLE = "Name: ___\n\nAllergies:\nnone\n\nChief Complaint:\npain\n"
WITHOUT_PREAMBLE = "Allergies:\n  none  \n"


@pytest.fixture
def lexicon(tmp_path: Path) -> Path:
    """Zapíše lexikon se dvěma sekcemi.

    Returns:
        Cesta k lexikonu v `tmp_path`.
    """
    path = tmp_path / "lexicon.json"
    document = {
        "sections": {
            "chief_complaint": {
                "label_cs": "Hlavní obtíže",
                "synonyms": ["Chief Complaint"],
            },
            "allergies": {"label_cs": "Alergie", "synonyms": ["Allergies"]},
        },
        "preamble": ["Name"],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_rows_follow_text_order_with_other_only_in_preamble(lexicon: Path) -> None:
    """Pozice jdou od 0 v pořadí textu a `other` má jen zpráva s preambulí."""
    result = note_sections(lexicon, ["a", "b"], [WITH_PREAMBLE, WITHOUT_PREAMBLE])

    assert set(result) == COLUMNS
    assert len({len(column) for column in result.values()}) == 1
    assert result["note_id"] == ["a", "a", "a", "b"]
    assert result["position"] == [0, 1, 2, 0]
    assert result["section"] == ["other", "allergies", "chief_complaint", "allergies"]


def test_span_sums_to_text_and_body_is_stripped(lexicon: Path) -> None:
    """Součet `span_chars` je délka zprávy, `body_chars` je tělo bez okrajových mezer."""
    result = note_sections(lexicon, ["a"], [WITH_PREAMBLE])
    assert sum(int(span) for span in result["span_chars"]) == len(WITH_PREAMBLE)

    body = note_sections(lexicon, ["b"], ["Allergies:\n  none \n"])
    assert body["body_chars"] == [4]


def test_text_without_headers_is_single_other_row(lexicon: Path) -> None:
    """Zpráva bez hlaviček je jeden řádek `other` přes celou délku."""
    text = "no headers here"
    result = note_sections(lexicon, ["x"], [text])

    assert result["section"] == ["other"]
    assert result["span_chars"] == [len(text)]


def test_repeated_header_does_not_open_second_row(lexicon: Path) -> None:
    """Platí pravidlo prvního výskytu parseru, opakovaná hlavička je jen text těla."""
    result = note_sections(lexicon, ["x"], ["Allergies:\nnone\nAllergies:\nlater\n"])

    assert result["section"] == ["allergies"]


def test_empty_input_keeps_all_columns(lexicon: Path) -> None:
    """Prázdný vstup vrátí všech pět sloupců prázdných, aby R mělo typy."""
    result = note_sections(lexicon, [], [])

    assert result == {column: [] for column in COLUMNS}


def test_mismatched_lengths_are_rejected(lexicon: Path) -> None:
    """Nesoulad `note_ids` a `texts` je chyba volajícího, ne tichý ořez."""
    with pytest.raises(ValueError, match="shorter"):
        note_sections(lexicon, ["a", "b"], [WITHOUT_PREAMBLE])


def test_missing_lexicon_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor lexikonu zastaví render doménovou výjimkou."""
    with pytest.raises(InvalidInputError):
        note_sections(tmp_path / "missing.json", ["a"], [WITHOUT_PREAMBLE])


def test_other_is_the_domain_constant() -> None:
    """R čte klíč `other` z modulu, takže nesmí být vlastní kopie."""
    assert OTHER is section_headers.OTHER
    assert eda_sections.OTHER == "other"


def test_repository_lexicon_loads() -> None:
    """Lexikon v repu projde vstupním bodem."""
    result = note_sections(REPOSITORY_SECTION_HEADERS, ["a"], [WITHOUT_PREAMBLE])

    assert result["section"] == ["allergies"]
