"""Ruční fixture F pro společný generátor syntetického MIMIC."""

from typing import Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.language import Language
from mediparse.domain.note_plan import NotePlan
from mediparse.domain.note_structure import Sex
from mediparse.domain.section_headers import LexiconSection, SectionLexicon
from mediparse.domain.synthetic_mimic import input_notes

TEXT: Final = """Name: ___
Chief Complaint:
Cough.

Brief Hospital Course:
Patient   improved.
Started insulin.

Medications on Admission:
Aspirin.
Metformin.

Discharge Diagnosis:
Diabetes.
"""
LEXICON: Final = SectionLexicon(
    sections={
        "chief_complaint": LexiconSection(
            label_cs="Obtíže", synonyms=("Chief Complaint",)
        ),
        "hospital_course": LexiconSection(
            label_cs="Průběh", synonyms=("Brief Hospital Course",)
        ),
        "medications_on_admission": LexiconSection(
            label_cs="Léky", synonyms=("Medications on Admission",)
        ),
        "discharge_diagnosis": LexiconSection(
            label_cs="Diagnóza", synonyms=("Discharge Diagnosis",), diagnostic=True
        ),
    },
    preamble=("Name",),
)


def _plan(
    note_id: str,
    subject: int,
    sex: Sex,
    labels: tuple[Diagnosis, ...],
    sections: tuple[str, ...],
) -> NotePlan:
    return NotePlan(
        note_id=note_id,
        subject_id=subject,
        sex=sex,
        labels=labels,
        sections=sections,
        age_marker=False,
        subheadings=(),
        section_words={},
        narrative_deid=0,
        mentions=(),
    )


PLANS: Final = (
    _plan(
        "90000001-DS-1",
        90000001,
        Sex.FEMALE,
        (Diagnosis.DIABETES,),
        tuple(LEXICON.sections),
    ),
    _plan(
        "90000001-DS-2",
        90000001,
        Sex.FEMALE,
        (),
        ("chief_complaint", "discharge_diagnosis"),
    ),
    _plan(
        "90000002-DS-1",
        90000002,
        Sex.MALE,
        (Diagnosis.COPD, Diagnosis.DIABETES),
        tuple(LEXICON.sections),
    ),
)
NOTES: Final = input_notes(
    PLANS,
    {
        p.note_id: TEXT
        if "hospital_course" in p.sections
        else "Chief Complaint:\nCough.\n\nDischarge Diagnosis:\nNone.\n"
        for p in PLANS
    },
    Language.EN,
)
