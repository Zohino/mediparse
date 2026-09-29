"""Plán zprávy: uspořádané labely, round-trip přes JSON a odmítnutí cizí struktury."""

from dataclasses import replace

import pytest

from mediparse.domain.labels import Diagnosis
from mediparse.domain.mentions import Mention, MentionStatus
from mediparse.domain.note_plan import MismatchedPlanPartsError, NotePlan, note_plan
from mediparse.domain.note_structure import NoteStructure, Sex
from mediparse.domain.synthetic_patients import SyntheticNote

NOTE_ID = "90000001-DS-1"
NOTE = SyntheticNote(
    NOTE_ID, 90000001, frozenset({Diagnosis.AKI, Diagnosis.DIABETES, Diagnosis.CKD})
)
STRUCTURE = NoteStructure(
    note_id=NOTE_ID,
    sex=Sex.MALE,
    age_marker=True,
    sections=("past_medical_history", "hospital_course", "discharge_diagnosis"),
    subheadings=("TRANSITIONAL ISSUES",),
    narrative_words=120,
    section_words=(("hospital_course", 120),),
    narrative_deid=4,
)
MENTIONS = (
    Mention(Diagnosis.DIABETES, MentionStatus.AFFIRMED, ("past_medical_history",)),
    Mention(Diagnosis.AKI, MentionStatus.AFFIRMED, ("discharge_diagnosis",)),
)


def test_labels_follow_diagnosis_order() -> None:
    """Labely z množiny se v plánu seřadí podle pořadí diagnóz, ne podle hashe."""
    plan = note_plan(NOTE, STRUCTURE, MENTIONS)

    assert plan.labels == (Diagnosis.DIABETES, Diagnosis.CKD, Diagnosis.AKI)


def test_plan_survives_json_round_trip() -> None:
    """Zapsaný plán se načte zpět beze změny, schéma tak validuje i čtení v S11d."""
    plan = note_plan(NOTE, STRUCTURE, MENTIONS)

    assert NotePlan.model_validate_json(plan.model_dump_json()) == plan


def test_structure_of_other_note_is_rejected() -> None:
    """Struktura cizí zprávy se do plánu nesloží."""
    other = replace(STRUCTURE, note_id="90000002-DS-1")

    with pytest.raises(MismatchedPlanPartsError):
        note_plan(NOTE, other, MENTIONS)
