"""Plán zprávy: uspořádané labely, round-trip přes JSON a odmítnutí cizí struktury."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.mentions import Mention, MentionStatus
from mediparse.domain.note_plan import MismatchedPlanPartsError, NotePlan, note_plan
from mediparse.domain.note_structure import NoteStructure, Sex
from mediparse.domain.synthetic_patients import SyntheticNote

NOTE_ID = "90000001-DS-1"
NOTE = SyntheticNote(
    NOTE_ID,
    90000001,
    frozenset({Diagnosis.PNEUMONIA, Diagnosis.DIABETES, Diagnosis.CKD}),
)
STRUCTURE = NoteStructure(
    note_id=NOTE_ID,
    sex=Sex.MALE,
    age_marker=True,
    sections=("past_medical_history", "hospital_course", "discharge_diagnosis"),
    subheadings=("TRANSITIONAL ISSUES",),
    section_words={"hospital_course": 120},
    narrative_deid=4,
)
MENTIONS = (
    Mention(
        diagnosis=Diagnosis.DIABETES,
        status=MentionStatus.AFFIRMED,
        sections=("past_medical_history",),
    ),
    Mention(
        diagnosis=Diagnosis.PNEUMONIA,
        status=MentionStatus.AFFIRMED,
        sections=("discharge_diagnosis",),
    ),
)


def test_labels_follow_diagnosis_order() -> None:
    """Labely z množiny se v plánu seřadí podle pořadí diagnóz, ne podle hashe."""
    plan = note_plan(NOTE, STRUCTURE, MENTIONS)

    assert plan.labels == (Diagnosis.DIABETES, Diagnosis.CKD, Diagnosis.PNEUMONIA)


def test_plan_survives_json_round_trip() -> None:
    """Zapsaný plán se načte zpět beze změny, schéma tak validuje i čtení v S11d."""
    plan = note_plan(NOTE, STRUCTURE, MENTIONS)

    assert NotePlan.model_validate_json(plan.model_dump_json()) == plan


def test_structure_of_other_note_is_rejected() -> None:
    """Struktura cizí zprávy se do plánu nesloží."""
    other = replace(STRUCTURE, note_id="90000002-DS-1")

    with pytest.raises(MismatchedPlanPartsError):
        note_plan(NOTE, other, MENTIONS)


def test_plan_of_other_patient_than_note_id_is_rejected() -> None:
    """Subject_id, který neodpovídá note_id, plán odmítne, ať vznikl kudy chce."""
    plan = note_plan(NOTE, STRUCTURE, MENTIONS)

    with pytest.raises(ValidationError, match="nepatří pacientovi"):
        NotePlan.model_validate_json(
            plan.model_copy(update={"subject_id": 90000002}).model_dump_json()
        )


def test_plan_with_malformed_note_id_is_rejected() -> None:
    """Jméno mimo skladbu note_id plán odmítne."""
    plan = note_plan(NOTE, STRUCTURE, MENTIONS)

    with pytest.raises(ValidationError, match="Neplatné note_id"):
        NotePlan.model_validate_json(
            plan.model_copy(update={"note_id": "n1"}).model_dump_json()
        )
