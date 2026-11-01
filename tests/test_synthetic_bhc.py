"""Target BHC má ruční orákulum nezávislé na parseru sekcí."""

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import LexiconSection
from mediparse.domain.synthetic_bhc import bhc_target
from tests.synthetic_support import LEXICON, PLANS, TEXT


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
