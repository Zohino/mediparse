"""Lexikon sekcí, normalizace hlavičky, rozpoznání hlaviček, rozdělení zprávy na sekce a strip diagnostických sekcí."""

from __future__ import annotations

import json
import re
from typing import Any, Final

import pytest
from pydantic import ValidationError

from mediparse.domain.language import Language
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
    strip_diagnostic_sections,
)
from mediparse.infrastructure.sampler_config import load_sampler_config
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory
from tests.section_support import (
    MIMIC_SHAPED_NOTES,
    REPOSITORY_LEXICON,
    mimic_shaped_notes,
)
from tests.support import REPOSITORY_CONFIG, REPOSITORY_CORPUS


def _lexicon_json(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "sections": {
            "chief_complaint": {
                "label_cs": "Hlavní obtíže",
                "synonyms": ["Chief Complaint"],
            },
            "allergies": {
                "label_cs": "Alergie",
                "synonyms": ["Allergies"],
                "diagnostic": True,
            },
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


def test_section_without_flag_is_not_diagnostic() -> None:
    """Chybějící pole `diagnostic` znamená, že sekce diagnostická není."""
    sections = _lexicon().sections

    assert not sections["chief_complaint"].diagnostic
    assert sections["allergies"].diagnostic


def test_repository_lexicon_marks_only_discharge_diagnosis_as_diagnostic() -> None:
    """Čísla sešitu 9 stojí na tom, že strip odstraňuje právě tuto sekci."""
    assert REPOSITORY_LEXICON.diagnostic_sections == {"discharge_diagnosis"}


def test_synthetic_diagnosis_section_is_removed_by_strip() -> None:
    """Zmínka diagnózy syntetiky leží v sekci, kterou ablace odstraní."""
    plan = load_sampler_config(REPOSITORY_CONFIG)

    assert plan.mentions.diagnosis_section in REPOSITORY_LEXICON.diagnostic_sections


def test_minimal_lexicon_is_accepted() -> None:
    """Lexikon se sekcemi a preambulí projde schématem."""
    assert set(_lexicon().sections) == {"chief_complaint", "allergies"}


def _section(
    *synonyms: str, label_cs: str = "A", diagnostic: bool = False
) -> dict[str, object]:
    return {
        "label_cs": label_cs,
        "synonyms": list(synonyms),
        "diagnostic": diagnostic,
    }


INVALID_LEXICONS: Final = {
    "sekce bez synonym": ({"sections": {"a": _section()}}, "at least 1 item"),
    "prázdné label_cs": (
        {"sections": {"a": _section("Allergies", label_cs="")}},
        "at least 1 character",
    ),
    "bez sekcí": ({"sections": {}}, "at least 1 item"),
    "kolize dvou sekcí po normalizaci": (
        {
            "sections": {
                "a": _section("Allergies", diagnostic=True),
                "b": _section("ALLERGIES.", diagnostic=True),
            }
        },
        "splývají hlavičky: allergies",
    ),
    "kolize sekce s preamble": (
        {"preamble": ["Name", "ALLERGIES."]},
        "splývají hlavičky: allergies",
    ),
    "synonymum s číslicí": (
        {"sections": {"a": _section("Type 2 Diabetes", diagnostic=True)}},
        "nesplňuje výraz hlavičky",
    ),
    "malé první písmeno": (
        {"sections": {"a": _section("allergies", diagnostic=True)}},
        "nesplňuje výraz hlavičky",
    ),
    "odsazené synonymum": (
        {"sections": {"a": _section(" Allergies", diagnostic=True)}},
        "nesplňuje výraz hlavičky",
    ),
    "normalizovaný tvar pod 3 znaky": (
        {"sections": {"a": _section("Ab", diagnostic=True)}},
        "kratší než",
    ),
    "pole preambule mimo výraz hlavičky": (
        {"preamble": ["attending"]},
        "Pole 'attending' nesplňuje výraz hlavičky",
    ),
    "pole preambule pod 3 znaky": ({"preamble": ["Id"]}, "Pole 'Id' má"),
    "vyhrazený klíč other": (
        {"sections": {"other": _section("Allergies", diagnostic=True)}},
        "vyhrazený klíč: other",
    ),
    "neznámý klíč na nejvyšší úrovni": ({"extra": 1}, "Extra inputs"),
    "žádná diagnostická sekce": (
        {"sections": {"a": _section("Allergies")}},
        "aspoň jednu diagnostickou sekci",
    ),
    "příznak diagnostic jako řetězec": (
        {"sections": {"a": {**_section("Allergies"), "diagnostic": "true"}}},
        "Input should be a valid boolean",
    ),
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
        _lexicon(sections={"a": _section(synonym, diagnostic=True)})


DISCHARGE_LEXICON = SectionLexicon.model_validate_json(
    json.dumps({
        "sections": {
            "allergies": _section("Allergies"),
            "chief_complaint": _section("Chief Complaint", label_cs="Obtíže"),
            "past_medical_history": _section("Past Medical History"),
            "discharge_diagnosis": _section(
                "Discharge Diagnosis", "Discharge Diagnoses", diagnostic=True
            ),
        },
        "preamble": ["Name", "Unit No", "Service", "Attending"],
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

    assert find_section_headers(text, DISCHARGE_LEXICON) == (
        SectionHeader("past_medical_history", 0, 21),
        SectionHeader("chief_complaint", 26, 42),
    )


def test_repeat_through_other_synonym_returns_first_only() -> None:
    """Různá synonyma téže sekce se počítají jako jedna sekce."""
    text = "Discharge Diagnosis:\nCHF\nDischarge Diagnoses:\nCOPD"

    assert _found(text) == [("discharge_diagnosis", "Discharge Diagnosis:")]


def _synthetic_notes() -> dict[str, str]:
    notes = CorpusDirectory(REPOSITORY_CORPUS).notes(Language.EN)
    assert notes
    return notes


def test_synthetic_corpus_matches_generator_headers() -> None:
    """Hlavičky i pozice odpovídají tomu, co do zprávy umístil generátor."""
    headers = {
        key: header
        for key, header in load_sampler_config(
            REPOSITORY_CONFIG
        ).structure.headers.items()
        if key in REPOSITORY_LEXICON.sections
    }
    pattern = line_headers(headers.values())
    by_header = {header: key for key, header in headers.items()}

    for note_id, text in _synthetic_notes().items():
        expected = tuple(
            SectionHeader(by_header[m["header"]], m.start(), m.end())
            for m in pattern.finditer(text)
        )
        assert find_section_headers(text, REPOSITORY_LEXICON) == expected, note_id


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
    "pole za sekcí ji ukončí jako other (šablona MIMIC)": (
        (
            "Service: MEDICINE\nAllergies:\nPenicillin\nAttending: ___.\n"
            "Chief Complaint:\npain"
        ),
        (
            NoteSection(OTHER, "", "Service: MEDICINE\n"),
            NoteSection("allergies", "Allergies:", "\nPenicillin\n"),
            NoteSection(OTHER, "", "Attending: ___.\n"),
            NoteSection("chief_complaint", "Chief Complaint:", "\npain"),
        ),
    ),
    "pole za sebou dají jednu část other": (
        "Allergies:\nnone\n  Attending: ___\nService: MED\nChief Complaint:\nx",
        (
            NoteSection("allergies", "Allergies:", "\nnone\n"),
            NoteSection(OTHER, "", "  Attending: ___\nService: MED\n"),
            NoteSection("chief_complaint", "Chief Complaint:", "\nx"),
        ),
    ),
    "pole v preambuli se spotřebuje, opakování zůstane v těle": (
        "Attending: ___\nAllergies:\nnone\nAttending: Dr. ___\nChief Complaint:\nx",
        (
            NoteSection(OTHER, "", "Attending: ___\n"),
            NoteSection("allergies", "Allergies:", "\nnone\nAttending: Dr. ___\n"),
            NoteSection("chief_complaint", "Chief Complaint:", "\nx"),
        ),
    ),
    "opakované pole za sekcí zůstane v těle": (
        "Allergies:\nnone\nAttending: ___\nChief Complaint:\nAttending: ___\nx",
        (
            NoteSection("allergies", "Allergies:", "\nnone\n"),
            NoteSection(OTHER, "", "Attending: ___\n"),
            NoteSection("chief_complaint", "Chief Complaint:", "\nAttending: ___\nx"),
        ),
    ),
    "pole uprostřed řádku ani mimo lexikon sekci neukončí": (
        "Allergies:\nnone  Attending: ___\nUnit: 4\nChief Complaint:\nx",
        (
            NoteSection("allergies", "Allergies:", "\nnone  Attending: ___\nUnit: 4\n"),
            NoteSection("chief_complaint", "Chief Complaint:", "\nx"),
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


@pytest.mark.parametrize(
    "text",
    [case[0] for case in SPLIT_CASES.values()],
    ids=SPLIT_CASES.keys(),
)
def test_split_sections_compose_back_to_original(text: str) -> None:
    """Hlavičky a těla dají v pořadí přesně původní text."""
    sections = split_sections(text, DISCHARGE_LEXICON)

    assert "".join(s.header + s.body for s in sections) == text


def test_split_synthetic_corpus_composes_originals() -> None:
    """Každá syntetická zpráva se rozpadne na sekce lexikonu a složí se zpět."""
    for note_id, text in _synthetic_notes().items():
        sections = split_sections(text, REPOSITORY_LEXICON)
        found = find_section_headers(text, REPOSITORY_LEXICON)

        assert "".join(s.header + s.body for s in sections) == text, note_id
        assert [s.section for s in sections] == [OTHER, *(h.section for h in found)], (
            note_id
        )


def test_split_sections_ends_allergies_at_repository_preamble_field() -> None:
    """Pole Attending z lexikonu v config/ ukončí tělo alergií jako v šabloně MIMIC."""
    text = "Allergies:\nPenicillin\nAttending: ___.\nChief Complaint:\npain"

    sections = split_sections(text, REPOSITORY_LEXICON)

    assert [(s.section, s.body) for s in sections] == [
        ("allergies", "\nPenicillin\n"),
        (OTHER, "Attending: ___.\n"),
        ("chief_complaint", "\npain"),
    ]


def test_split_synthetic_corpus_keeps_surgical_history_in_medical_history() -> None:
    """Past Surgical History není hlavička lexikonu a zůstává v těle předchozí sekce."""
    surgical = "Past Surgical History:"
    checked = 0

    for note_id, text in _synthetic_notes().items():
        if surgical not in text:
            continue
        bodies = {s.section: s.body for s in split_sections(text, REPOSITORY_LEXICON)}
        assert surgical in bodies["past_medical_history"], note_id
        checked += 1

    assert checked


MIMIC_SHAPED_EXPECTED: Final = json.loads(
    (MIMIC_SHAPED_NOTES / "expected.json").read_text(encoding="utf-8")
)
MIN_MIMIC_SHAPED_NOTES: Final = 12


def test_mimic_shaped_notes_match_expectations() -> None:
    """Každá zpráva má očekávání a každé očekávání zprávu; sada má aspoň 12 zpráv."""
    notes = mimic_shaped_notes()

    assert notes.keys() == MIMIC_SHAPED_EXPECTED.keys()
    assert len(notes) >= MIN_MIMIC_SHAPED_NOTES


def _with_first_lines(
    text: str, sections: tuple[NoteSection, ...]
) -> list[list[object]]:
    """Klíč, hlavička a číslo řádku (od 1), kde část začíná.

    Returns:
        Trojice ve tvaru `expected.json`.
    """
    result: list[list[object]] = []
    offset = 0
    for section in sections:
        # Část začíná na začátku řádku; cut uvnitř řádku ukáže jen sloupec, ne číslo řádku.
        assert offset == 0 or text[offset - 1] == "\n"
        result.append([section.section, section.header, text[:offset].count("\n") + 1])
        offset += len(section.header) + len(section.body)
    return result


@pytest.mark.parametrize("name", MIMIC_SHAPED_EXPECTED, ids=MIMIC_SHAPED_EXPECTED)
def test_mimic_shaped_note_splits_as_expected(name: str) -> None:
    """Celá zpráva ve tvaru šablony MIMIC se rozdělí podle očekávání a složí se zpět."""
    text = mimic_shaped_notes()[name]

    sections = split_sections(text, REPOSITORY_LEXICON)

    assert _with_first_lines(text, sections) == MIMIC_SHAPED_EXPECTED[name]["sections"]
    assert "".join(s.header + s.body for s in sections) == text


STRIP_CASES: Final = {
    "sekce uprostřed zmizí s hlavičkou a koncovým řádkem": (
        "Allergies:\nnone\nDischarge Diagnosis:\nCHF\nChief Complaint:\npain",
        "Allergies:\nnone\nChief Complaint:\npain",
    ),
    "sekce na konci bez koncového newline": (
        "Allergies:\nnone\nDischarge Diagnosis:\nCHF",
        "Allergies:\nnone\n",
    ),
    "hodnota na řádku hlavičky": (
        "Allergies:\nnone\nDischarge Diagnosis: CHF\nChief Complaint:\npain",
        "Allergies:\nnone\nChief Complaint:\npain",
    ),
    "odsazená hlavička a varianta synonyma": (
        "Allergies:\nnone\n  DISCHARGE DIAGNOSES.:\nCHF\nChief Complaint:\npain",
        "Allergies:\nnone\nChief Complaint:\npain",
    ),
    "zpráva bez sekce zůstane beze změny": (
        "Allergies:\nnone\nChief Complaint:\npain\n",
        "Allergies:\nnone\nChief Complaint:\npain\n",
    ),
    "prázdný text": ("", ""),
    "CRLF": (
        "Allergies:\r\nnone\r\nDischarge Diagnosis:\r\nCHF\r\nChief Complaint:\r\nx",
        "Allergies:\r\nnone\r\nChief Complaint:\r\nx",
    ),
    "pole preambule za sekcí zůstane": (
        "Discharge Diagnosis:\nCHF\nAttending: ___\nAllergies:\nnone",
        "Attending: ___\nAllergies:\nnone",
    ),
    "opakovaná hlavička v těle jiné sekce zůstane (strip není idempotentní)": (
        (
            "Chief Complaint:\nx\nDischarge Diagnosis:\nCHF\nPast Medical History:\n"
            "Discharge Diagnosis:\nCOPD"
        ),
        "Chief Complaint:\nx\nPast Medical History:\nDischarge Diagnosis:\nCOPD",
    ),
}


@pytest.mark.parametrize(
    ("text", "expected"), STRIP_CASES.values(), ids=STRIP_CASES.keys()
)
def test_strip_removes_diagnostic_section_with_its_header(
    text: str, expected: str
) -> None:
    """Diagnostická sekce zmizí celá a ostatní text zůstane beze změny po bajtech."""
    assert strip_diagnostic_sections(text, DISCHARGE_LEXICON) == expected


def test_strip_removes_every_diagnostic_section_of_lexicon() -> None:
    """Označených sekcí může být víc a všechny zmizí."""
    lexicon = SectionLexicon.model_validate_json(
        json.dumps({
            "sections": {
                "allergies": _section("Allergies", diagnostic=True),
                "chief_complaint": _section("Chief Complaint"),
                "final": _section("Final Diagnosis", diagnostic=True),
            },
            "preamble": [],
        })
    )
    text = "Allergies:\nnone\nChief Complaint:\npain\nFinal Diagnosis:\nCHF\n"

    assert strip_diagnostic_sections(text, lexicon) == "Chief Complaint:\npain\n"


def _lines(text: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", text)


def _without_diagnostic_lines(text: str, sections: list[list[Any]]) -> str:
    """Orákulum stripu z `expected.json`: řádky od začátku diagnostické části po další část.

    Returns:
        Text bez těchto řádků.
    """
    lines = _lines(text)
    diagnostic = REPOSITORY_LEXICON.diagnostic_sections
    kept = list(lines)
    for index, (key, _header, first) in enumerate(sections):
        if key not in diagnostic:
            continue
        end = sections[index + 1][2] if index + 1 < len(sections) else len(lines) + 1
        for line_number in range(first, end):
            kept[line_number - 1] = ""
    return "".join(kept)


@pytest.mark.parametrize("name", MIMIC_SHAPED_EXPECTED, ids=MIMIC_SHAPED_EXPECTED)
def test_strip_mimic_shaped_note_matches_oracle_from_expectations(name: str) -> None:
    """Strip odpovídá textu bez řádků diagnostické části podle `expected.json`, ne podle parseru."""
    text = mimic_shaped_notes()[name]
    sections = MIMIC_SHAPED_EXPECTED[name]["sections"]

    stripped = strip_diagnostic_sections(text, REPOSITORY_LEXICON)

    assert stripped == _without_diagnostic_lines(text, sections)


def test_strip_mimic_shaped_notes_include_notes_with_and_without_section() -> None:
    """Ochrana proti vakuu: sada má zprávy se sekcí i bez ní a strip se projeví jen u prvních."""
    changed = {
        name
        for name, text in mimic_shaped_notes().items()
        if strip_diagnostic_sections(text, REPOSITORY_LEXICON) != text
    }

    assert changed
    assert changed != set(mimic_shaped_notes())
    assert not changed & {"headerless_addendum", "empty_sections"}
