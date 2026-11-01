"""Target BHC má ruční orákulum nezávislé na parseru sekcí."""

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import LexiconSection
from mediparse.domain.synthetic_bhc import bhc_rows, bhc_target
from tests.synthetic_support import (
    EDA_NOTES,
    EDA_PLANS,
    LEXICON,
    PLANS,
    TEXT,
)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (0, "Patient improved. Started insulin."),
        (
            1,
            "Patient improved. Started insulin. Medications on Admission: Aspirin. Metformin.",
        ),
        (2, "Patient improved."),
        (3, "Patient improved. Started insulin. Medications on Admission: Aspirin."),
    ],
)
def test_manual_bhc_cuts(variant: int, expected: str) -> None:
    """A1: všechny čtyři řezy odpovídají ručnímu výpočtu z textu F."""
    assert bhc_target(PLANS[0], TEXT, LEXICON, variant) == expected


def test_plan_without_bhc_has_no_target() -> None:
    """Chybějící sekce v plánu znamená chybějící řádek targetu."""
    assert bhc_target(PLANS[1], TEXT, LEXICON, 0) is None


@pytest.mark.parametrize(
    "header",
    ["Brief Hospital Course", "Medications on Admission", "Discharge Diagnosis"],
)
def test_missing_promised_header_is_refused(header: str) -> None:
    """Slíbená hlavička musí v textu existovat i na koncové hranici."""
    with pytest.raises(InvalidInputError, match="hlavičku"):
        bhc_target(PLANS[0], TEXT.replace(f"{header}:", "Missing:"), LEXICON, 0)


def test_whole_line_synonym_is_used() -> None:
    """Zmínka názvu v těle není hranicí; hlavička může použít jiné synonymum."""
    sections = dict(LEXICON.sections)
    sections["hospital_course"] = LexiconSection(
        label_cs="Průběh", synonyms=("Brief Hospital Course", "Hospital Course")
    )
    lexicon = LEXICON.model_copy(update={"sections": sections})
    text = TEXT.replace("Cough.", "Cough. Brief Hospital Course:").replace(
        "\nBrief Hospital Course:", "\nHospital Course:"
    )
    assert (
        bhc_target(PLANS[0], text, lexicon, 0) == "Patient improved. Started insulin."
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (0, "Recovered. Home."),
        (1, "Recovered. Home."),
        (2, "Recovered."),
        (3, "Recovered. Home."),
    ],
)
def test_bhc_as_last_section(variant: int, expected: str) -> None:
    """Poslední BHC končí až koncem zprávy, první řádek vybírá tělo bez úvodní mezery."""
    plan = PLANS[0].model_copy(
        update={"sections": ("chief_complaint", "hospital_course")}
    )
    text = "Chief Complaint:\nCough.\n\nBrief Hospital Course:\n\nRecovered.\nHome.\n"
    assert bhc_target(plan, text, LEXICON, variant) == expected


def test_following_section_at_end_of_note() -> None:
    """Celá následující poslední sekce patří do rozšířeného targetu."""
    plan = PLANS[0].model_copy(
        update={"sections": ("hospital_course", "medications_on_admission")}
    )
    text = "Brief Hospital Course:\nImproved.\n\nMedications on Admission:\nAspirin.\nMetformin.\n"
    assert (
        bhc_target(plan, text, LEXICON, 1)
        == "Improved. Medications on Admission: Aspirin. Metformin."
    )


def test_indented_headers_with_trailing_whitespace() -> None:
    """Odsazení a koncové mezery nejsou součástí názvu hlavičky."""
    text = (
        TEXT
        .replace("Brief Hospital Course:", "  Brief Hospital Course: \t")
        .replace("Medications on Admission:", "\tMedications on Admission: ")
        .replace("Discharge Diagnosis:", "  Discharge Diagnosis:  ")
    )
    assert (
        bhc_target(PLANS[0], text, LEXICON, 1)
        == "Patient improved. Started insulin. Medications on Admission: Aspirin. Metformin."
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (2, "Patient improved."),
        (3, "Patient improved. Started insulin. Medications on Admission: Aspirin."),
    ],
)
def test_blank_lines_do_not_replace_first_content_line(
    variant: int, expected: str
) -> None:
    """První řádek těla i další sekce znamená první neprázdný řádek."""
    text = TEXT.replace(
        "Brief Hospital Course:\n", "Brief Hospital Course:\n\n"
    ).replace("Medications on Admission:\n", "Medications on Admission:\n \n")
    assert bhc_target(PLANS[0], text, LEXICON, variant) == expected


@pytest.mark.parametrize("header", ["Medications on Admission", "Discharge Diagnosis"])
def test_following_headers_before_bhc_are_ignored(header: str) -> None:
    """Dřívější výskyt následující hlavičky neukončuje BHC zpětně."""
    text = f"{header}:\nEarlier section.\n\n{TEXT}"
    assert (
        bhc_target(PLANS[0], text, LEXICON, 1)
        == "Patient improved. Started insulin. Medications on Admission: Aspirin. Metformin."
    )


def test_plan_section_missing_from_lexicon_is_refused() -> None:
    """Chybějící položka lexikonu se hlásí doménovou výjimkou."""
    lexicon = LEXICON.model_copy(
        update={
            "sections": {
                key: section
                for key, section in LEXICON.sections.items()
                if key != "medications_on_admission"
            }
        }
    )
    with pytest.raises(InvalidInputError, match=r"Lexikon.*medications_on_admission"):
        bhc_target(PLANS[0], TEXT, lexicon, 0)


def test_empty_target_is_refused() -> None:
    """Prázdné tělo BHC není platný prázdný target."""
    text = "Brief Hospital Course:\n \nMedications on Admission:\nAspirin.\n\nDischarge Diagnosis:\nDiabetes.\n"
    with pytest.raises(InvalidInputError, match="Prázdný target BHC"):
        bhc_target(PLANS[0], text, LEXICON, 0)


def test_manual_bhc_rows() -> None:
    """A4: varianta se počítá jen mezi zprávami s BHC v pořadí note_id."""
    discharge = [{"note_id": note.note_id, "text": note.text} for note in EDA_NOTES]
    targets = bhc_rows(tuple(reversed(EDA_PLANS)), list(reversed(discharge)), LEXICON)
    assert targets == [
        {"note_id": "90000001-DS-1", "target": "Patient improved. Started insulin."},
        {
            "note_id": "90000002-DS-1",
            "target": "Patient improved. Started insulin. Medications on Admission: Aspirin. Metformin.",
        },
    ]


def test_bhc_row_without_plan_is_refused() -> None:
    """Zpráva bez protějšku v plánech má doménovou diagnostiku."""
    discharge = [{"note_id": "synthetic-unplanned", "text": TEXT}]
    with pytest.raises(
        InvalidInputError, match=r"Zprávy bez plánu: synthetic-unplanned\."
    ):
        bhc_rows((), discharge, LEXICON)


def test_bhc_rows_cycle_through_all_cuts() -> None:
    """Pět BHC s jednou zprávou bez BHC mezi nimi projde variantami 0, 1, 2, 3, 0."""
    plans = (
        PLANS[0],
        PLANS[1],
        *(
            PLANS[0].model_copy(update={"note_id": f"90000001-DS-{seq}"})
            for seq in range(3, 7)
        ),
    )
    discharge = [{"note_id": plan.note_id, "text": TEXT} for plan in reversed(plans)]
    assert bhc_rows(tuple(reversed(plans)), discharge, LEXICON) == [
        {"note_id": "90000001-DS-1", "target": "Patient improved. Started insulin."},
        {
            "note_id": "90000001-DS-3",
            "target": "Patient improved. Started insulin. Medications on Admission: Aspirin. Metformin.",
        },
        {"note_id": "90000001-DS-4", "target": "Patient improved."},
        {
            "note_id": "90000001-DS-5",
            "target": "Patient improved. Started insulin. Medications on Admission: Aspirin.",
        },
        {"note_id": "90000001-DS-6", "target": "Patient improved. Started insulin."},
    ]
