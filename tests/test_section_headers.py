"""Lexikon sekcí, normalizace hlavičky a rozpoznání hlaviček v textu zprávy."""

from __future__ import annotations

import json
from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.note_text import line_headers
from mediparse.domain.section_headers import (
    HEADER_PATTERN,
    OTHER,
    NoteSection,
    SectionHeader,
    SectionLexicon,
    find_section_headers,
    normalized,
    split_sections,
)
from mediparse.infrastructure.sampler_config import load_sampler_config
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from tests.support import (
    REPOSITORY_CONFIG,
    REPOSITORY_CORPUS,
    REPOSITORY_SECTION_HEADERS,
)


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
    "sekce bez synonym": ({"sections": {"a": _section()}}, "at least 1 item"),
    "prázdné label_cs": (
        {"sections": {"a": _section("Allergies", label_cs="")}},
        "at least 1 character",
    ),
    "bez sekcí": ({"sections": {}}, "at least 1 item"),
    "kolize dvou sekcí po normalizaci": (
        {"sections": {"a": _section("Allergies"), "b": _section("ALLERGIES.")}},
        "splývají hlavičky: allergies",
    ),
    "kolize sekce s preamble": (
        {"preamble": ["Name", "allergies"]},
        "splývají hlavičky: allergies",
    ),
    "synonymum s číslicí": (
        {"sections": {"a": _section("Type 2 Diabetes")}},
        "nesplňuje výraz hlavičky",
    ),
    "malé první písmeno": (
        {"sections": {"a": _section("allergies")}},
        "nesplňuje výraz hlavičky",
    ),
    "odsazené synonymum": (
        {"sections": {"a": _section(" Allergies")}},
        "nesplňuje výraz hlavičky",
    ),
    "normalizovaný tvar pod 3 znaky": (
        {"sections": {"a": _section("Ab")}},
        "kratší než",
    ),
    "vyhrazený klíč other": (
        {"sections": {"other": _section("Allergies")}},
        "vyhrazený klíč: other",
    ),
    "neznámý klíč na nejvyšší úrovni": ({"extra": 1}, "Extra inputs"),
}


@pytest.mark.parametrize(
    ("overrides", "message"), INVALID_LEXICONS.values(), ids=INVALID_LEXICONS.keys()
)
def test_invalid_lexicon_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    """Každé porušení pravidla lexikonu zastaví validaci se správným důvodem."""
    with pytest.raises(ValidationError, match=message):
        _lexicon(**overrides)


@pytest.mark.parametrize(
    ("text", "headers"),
    [
        ("Time 10:30", []),
        ("# CHF: yes", []),
        ("see Chief Complaint: x", []),
        ("  Chief Complaint:\n\tAllergies: none", ["Chief Complaint", "Allergies"]),
        (
            "Allergies:\r\nPenicillin\r\nSocial History:\r\nnone",
            ["Allergies", "Social History"],
        ),
        ("Service: MEDICINE", ["Service"]),
    ],
)
def test_header_pattern_mirrors_note_headers_cases(
    text: str, headers: list[str]
) -> None:
    """Případy note_headers z eda/tests/testthat/test-sections.R dávají totéž."""
    assert [m["header"] for m in HEADER_PATTERN.finditer(text)] == headers


@pytest.mark.parametrize("synonym", ["Ab", "Ab.", "Ab "])
def test_too_short_synonym_is_rejected(synonym: str) -> None:
    """Délku, kterou R filtruje v note_headers, Python hlídá ve validaci lexikonu."""
    with pytest.raises(ValidationError, match="kratší"):
        _lexicon(sections={"a": _section(synonym)})


DISCHARGE_LEXICON = SectionLexicon.model_validate_json(
    json.dumps({
        "sections": {
            "chief_complaint": _section("Chief Complaint", label_cs="Obtíže"),
            "past_medical_history": _section("Past Medical History"),
            "discharge_diagnosis": _section(
                "Discharge Diagnosis", "Discharge Diagnoses"
            ),
        },
        "preamble": ["Name", "Unit No"],
    })
)


def _found(text: str) -> list[tuple[str, str]]:
    return [
        (header.section, text[header.start : header.end])
        for header in find_section_headers(text, DISCHARGE_LEXICON)
    ]


def test_headers_have_exact_positions_in_text_order() -> None:
    """Start je začátek řádku, end je pozice za dvojtečkou."""
    text = "Name: ___\nChief Complaint:\nchest pain\nPast Medical History:\nCHF\n"

    assert find_section_headers(text, DISCHARGE_LEXICON) == (
        SectionHeader("chief_complaint", start=10, end=26),
        SectionHeader("past_medical_history", start=38, end=59),
    )


def test_indented_header_starts_at_line_start() -> None:
    """Odsazení patří hlavičce."""
    assert _found("x\n  \tChief Complaint: pain") == [
        ("chief_complaint", "  \tChief Complaint:")
    ]


def test_crlf_and_last_line_without_newline() -> None:
    """Konec řádku CRLF a poslední řádek bez konce řádku nevadí."""
    text = "Chief Complaint:\r\npain\r\nDischarge Diagnosis:"

    assert [key for key, _ in _found(text)] == [
        "chief_complaint",
        "discharge_diagnosis",
    ]


def test_spelling_variant_maps_to_section_key() -> None:
    """Varianta mezer, velikosti a tečky se přiřadí ke klíči sekce."""
    assert _found("CHIEF  COMPLAINT :\nx\nPast Medical History.:\ny") == [
        ("chief_complaint", "CHIEF  COMPLAINT :"),
        ("past_medical_history", "Past Medical History.:"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "# Chief Complaint:",
        "see Chief Complaint: x",
        "IMAGING:\nVitals: 120",
        "Name: ___ Unit No: ___",
        "",
        "no headers at all",
    ],
)
def test_non_headers_are_skipped(text: str) -> None:
    """Krátké názvy, čas, předpona, uprostřed řádku, cizí podnadpisy a preamble."""
    assert find_section_headers(text, DISCHARGE_LEXICON) == ()


def test_inline_value_after_header_is_not_a_second_header() -> None:
    """Hodnota za hlavičkou na témže řádku hranici neotevře."""
    assert _found("Discharge Diagnosis: Primary: CHF") == [
        ("discharge_diagnosis", "Discharge Diagnosis:")
    ]


def test_repeated_header_stays_in_body() -> None:
    """Druhý Past Medical History: za jinou sekcí se nevrátí."""
    text = (
        "Past Medical History:\nCHF\nChief Complaint:\npain\nPast Medical History:\nold"
    )

    assert [key for key, _ in _found(text)] == [
        "past_medical_history",
        "chief_complaint",
    ]


def test_repeat_through_other_synonym_returns_first_only() -> None:
    """Různá synonyma téže sekce se počítají jako jedna sekce."""
    text = "Discharge Diagnosis:\nCHF\nDischarge Diagnoses:\nCOPD"

    assert _found(text) == [("discharge_diagnosis", "Discharge Diagnosis:")]


def test_synthetic_corpus_matches_generator_headers() -> None:
    """Hlavičky i pozice odpovídají tomu, co do zprávy umístil generátor."""
    lexicon = load_section_lexicon(REPOSITORY_SECTION_HEADERS)
    headers = {
        key: header
        for key, header in load_sampler_config(
            REPOSITORY_CONFIG
        ).structure.headers.items()
        if key in lexicon.sections
    }
    pattern = line_headers(headers.values())
    by_header = {header: key for key, header in headers.items()}
    notes = sorted((REPOSITORY_CORPUS / "en").glob("*.txt"))

    assert notes
    for note in notes:
        text = note.read_text(encoding="utf-8")
        expected = tuple(
            SectionHeader(by_header[m["header"]], m.start(), m.end())
            for m in pattern.finditer(text)
        )
        assert find_section_headers(text, lexicon) == expected, note.name


SPLIT_CASES: Final = {
    "preambule a tři sekce": (
        (
            "Name: ___\nChief Complaint:\nchest pain\nPast Medical History:\nCHF\n"
            "Discharge Diagnosis:\nCOPD"
        ),
        (
            NoteSection(OTHER, "", "Name: ___\n"),
            NoteSection("chief_complaint", "Chief Complaint:", "\nchest pain\n"),
            NoteSection("past_medical_history", "Past Medical History:", "\nCHF\n"),
            NoteSection("discharge_diagnosis", "Discharge Diagnosis:", "\nCOPD"),
        ),
    ),
    "bez hlavičky": ("just words", (NoteSection(OTHER, "", "just words"),)),
    "prázdný text": ("", ()),
    "začíná hlavičkou": (
        "Chief Complaint:\npain",
        (NoteSection("chief_complaint", "Chief Complaint:", "\npain"),),
    ),
    "opakovaná hlavička a synonymum zůstanou v těle": (
        (
            "Discharge Diagnosis:\nCHF\nChief Complaint:\nx\n"
            "Discharge Diagnoses:\nCOPD\nChief Complaint:\ny"
        ),
        (
            NoteSection("discharge_diagnosis", "Discharge Diagnosis:", "\nCHF\n"),
            NoteSection(
                "chief_complaint",
                "Chief Complaint:",
                "\nx\nDischarge Diagnoses:\nCOPD\nChief Complaint:\ny",
            ),
        ),
    ),
    "prázdná sekce": (
        "Chief Complaint:\nPast Medical History:\nCHF",
        (
            NoteSection("chief_complaint", "Chief Complaint:", "\n"),
            NoteSection("past_medical_history", "Past Medical History:", "\nCHF"),
        ),
    ),
    "odsazení, CRLF a poslední řádek bez konce": (
        "x\r\n  Chief Complaint:\r\npain\r\nDischarge Diagnosis:",
        (
            NoteSection(OTHER, "", "x\r\n"),
            NoteSection("chief_complaint", "  Chief Complaint:", "\r\npain\r\n"),
            NoteSection("discharge_diagnosis", "Discharge Diagnosis:", ""),
        ),
    ),
}


@pytest.mark.parametrize(
    ("text", "expected"), SPLIT_CASES.values(), ids=SPLIT_CASES.keys()
)
def test_split_sections_cuts_text_into_named_units(
    text: str, expected: tuple[NoteSection, ...]
) -> None:
    """Preambule je other, hlavička nese odsazení a tělo drží konec řádku."""
    assert split_sections(text, DISCHARGE_LEXICON) == expected


@pytest.mark.parametrize("text", [case[0] for case in SPLIT_CASES.values()])
def test_split_sections_compose_back_to_original(text: str) -> None:
    """Hlavičky a těla dají v pořadí přesně původní text."""
    sections = split_sections(text, DISCHARGE_LEXICON)

    assert "".join(s.header + s.body for s in sections) == text
