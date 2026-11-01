"""Tabulky EDA jsou navázané na plány, zprávy a pacienty korpusu."""

from collections import defaultdict
from datetime import datetime
from typing import Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note_structure import Sex
from mediparse.domain.synthetic_mimic import corpus_tables
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES
from tests.synthetic_support import LEXICON, NOTES, PLANS, repository_tables

EXPECTED_ADMISSIONS: Final = 3
DUA_THRESHOLD: Final = 10
EXPECTED_NOTES: Final = 400
EXPECTED_PATIENTS: Final = 167
EXPECTED_TARGETS: Final = 357


def test_manual_demography_and_bhc() -> None:
    """A2 až A4: dvě osoby, tři přijetí a dva nezávislé targety podle F."""
    tables = corpus_tables(
        PLANS, NOTES, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES), LEXICON
    )
    assert tables["patients"] == [
        {
            "subject_id": "90000001",
            "gender": "F",
            "anchor_age": "15",
            "anchor_year": "2150",
            "anchor_year_group": "2017 - 2019",
        },
        {
            "subject_id": "90000002",
            "gender": "M",
            "anchor_age": "25",
            "anchor_year": "2150",
            "anchor_year_group": "2008 - 2010",
        },
    ]
    admissions = tables["admissions"]
    assert len(admissions) == EXPECTED_ADMISSIONS
    assert {r["hadm_id"] for r in admissions} == {
        r["hadm_id"] for r in tables["discharge"]
    }
    assert [r["insurance"] for r in admissions] == ["Medicare", "Medicare", "Private"]
    assert all(r["admittime"] == "2155-01-01 00:00:00" for r in admissions)
    assert tables["mimic-iv-bhc"] == [
        {"note_id": "90000001-DS-1", "target": "Patient improved. Started insulin."},
        {
            "note_id": "90000002-DS-1",
            "target": "Patient improved. Started insulin. Medications on Admission: Aspirin. Metformin.",
        },
    ]


def test_dictionary_has_each_used_pair_once() -> None:
    """Číselník popisuje všechny použité kódy včetně výplně a obou verzí."""
    tables = corpus_tables(
        PLANS, NOTES, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES), LEXICON
    )
    assert tables["d_icd_diagnoses"] == [
        {"icd_code": "249", "icd_version": "9", "long_title": "diabetes ICD-9"},
        {"icd_code": "491", "icd_version": "9", "long_title": "copd ICD-9"},
        {"icd_code": "E08", "icd_version": "10", "long_title": "diabetes ICD-10"},
        {"icd_code": "V700", "icd_version": "9", "long_title": "filler ICD-9"},
        {"icd_code": "Z0000", "icd_version": "10", "long_title": "filler ICD-10"},
    ]


def test_repository_demography_has_at_least_ten_patients_per_value() -> None:
    """A6: každá demografická hodnota a verze ICD má aspoň deset různých pacientů."""
    tables = repository_tables()
    for table, columns in {
        "patients": ("gender", "anchor_age", "anchor_year_group"),
        "admissions": ("insurance", "marital_status", "race", "language"),
        "diagnoses_icd": ("icd_version",),
    }.items():
        for column in columns:
            subjects = defaultdict(set)
            for row in tables[table]:
                subjects[row[column]].add(row["subject_id"])
            assert subjects
            assert all(len(group) >= DUA_THRESHOLD for group in subjects.values()), (
                table,
                column,
                subjects,
            )
    assert len(tables["discharge"]) == EXPECTED_NOTES
    assert len(tables["patients"]) == EXPECTED_PATIENTS
    assert len(tables["mimic-iv-bhc"]) == EXPECTED_TARGETS


@pytest.mark.parametrize("missing", ["plan", "note"])
def test_unpaired_plans_and_notes_are_refused(missing: str) -> None:
    """Přímý volající korpusové vrstvy dostane doménovou chybu pro obě neshody."""
    plans = PLANS[:-1] if missing == "plan" else PLANS
    notes = NOTES[:-1] if missing == "note" else NOTES
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    with pytest.raises(InvalidInputError, match=r"plán|zpráv"):
        corpus_tables(plans, notes, codes, LEXICON)


def test_conflicting_patient_sexes_are_refused() -> None:
    """Jedna osoba nemůže mít v plánech rozdílné pohlaví závislé na pořadí zpráv."""
    plans = (PLANS[0], PLANS[1].model_copy(update={"sex": Sex.MALE}), PLANS[2])
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    with pytest.raises(InvalidInputError, match=r"90000001.*pohlaví"):
        corpus_tables(plans, NOTES, codes, LEXICON)


def test_discharge_is_not_before_admission() -> None:
    """Zpráva vznikne při přijetí nebo po něm, u všech přijetí fixture F."""
    tables = corpus_tables(
        PLANS, NOTES, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES), LEXICON
    )
    admissions = {
        row["hadm_id"]: datetime.fromisoformat(row["admittime"])
        for row in tables["admissions"]
    }
    for row in tables["discharge"]:
        charttime = datetime.fromisoformat(row["charttime"])
        storetime = datetime.fromisoformat(row["storetime"])
        assert charttime >= admissions[row["hadm_id"]]
        assert storetime >= charttime
