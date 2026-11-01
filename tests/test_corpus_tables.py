"""Tabulky EDA jsou navázané na plány, zprávy a pacienty korpusu."""

from datetime import datetime
from typing import Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.note_structure import Sex
from mediparse.domain.synthetic_mimic import corpus_tables, input_notes
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES
from tests.synthetic_support import (
    EDA_NOTES,
    EDA_PLANS,
    NOTES,
    PLANS,
    repository_tables,
    with_coverage_groups,
)

EXPECTED_ADMISSIONS: Final = 33
EXPECTED_NOTES: Final = 400
EXPECTED_PATIENTS: Final = 167
EXPECTED_MASKED_NOTES: Final = 38


def test_manual_age_cycle() -> None:
    """Osm pacientů pokryje věková pásma včetně maskovaného věku 91."""
    plans = tuple(
        PLANS[0].model_copy(
            update={
                "note_id": f"{90000001 + index}-DS-1",
                "subject_id": 90000001 + index,
            }
        )
        for index in range(8)
    )
    notes = input_notes(plans, {p.note_id: NOTES[0].text for p in plans}, Language.EN)
    plans, notes = with_coverage_groups(plans, notes)
    tables = corpus_tables(
        plans, notes, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    )
    assert [row["anchor_age"] for row in tables["patients"]] == [
        "15",
        "25",
        "35",
        "45",
        "55",
        "65",
        "75",
        "91",
    ]


def test_repository_masked_age_count() -> None:
    """Nezávislý součet z plánů při cyklu osmi věků dává 38 zpráv."""
    tables = repository_tables()
    ages = {row["anchor_age"] for row in tables["patients"]}
    assert "86" not in ages
    assert "91" in ages
    masked = {
        row["subject_id"] for row in tables["patients"] if row["anchor_age"] == "91"
    }
    assert (
        sum(row["subject_id"] in masked for row in tables["discharge"])
        == EXPECTED_MASKED_NOTES
    )


def test_manual_demography() -> None:
    """A2 až A3: dvě osoby F i s přidanými skupinami pokrytí."""
    tables = corpus_tables(
        EDA_PLANS,
        EDA_NOTES,
        load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES),
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
    assert [r["insurance"] for r in admissions] == ["Medicare"] * 32 + ["Private"]
    assert all(r["admittime"] == "2155-01-01 00:00:00" for r in admissions)


def test_dictionary_has_each_used_pair_once() -> None:
    """Číselník popisuje všechny použité kódy včetně výplně a obou verzí."""
    tables = corpus_tables(
        EDA_PLANS,
        EDA_NOTES,
        load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES),
    )
    assert tables["d_icd_diagnoses"] == [
        {"icd_code": "249", "icd_version": "9", "long_title": "diabetes ICD-9"},
        {"icd_code": "491", "icd_version": "9", "long_title": "copd ICD-9"},
        {"icd_code": "E08", "icd_version": "10", "long_title": "diabetes ICD-10"},
        {"icd_code": "J12", "icd_version": "10", "long_title": "pneumonia ICD-10"},
        {"icd_code": "V700", "icd_version": "9", "long_title": "filler ICD-9"},
        {"icd_code": "Z0000", "icd_version": "10", "long_title": "filler ICD-10"},
    ]


def test_repository_table_counts() -> None:
    """Korpusové tabulky zachovávají počty zpráv a pacientů; target skládá volající."""
    tables = repository_tables()
    assert len(tables["discharge"]) == EXPECTED_NOTES
    assert len(tables["patients"]) == EXPECTED_PATIENTS
    assert "mimic-iv-bhc" not in corpus_tables(
        EDA_PLANS, EDA_NOTES, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    )


@pytest.mark.parametrize("missing", ["plan", "note"])
def test_unpaired_plans_and_notes_are_refused(missing: str) -> None:
    """Přímý volající korpusové vrstvy dostane doménovou chybu pro obě neshody."""
    plans = PLANS[:-1] if missing == "plan" else PLANS
    notes = NOTES[:-1] if missing == "note" else NOTES
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    with pytest.raises(InvalidInputError, match=r"plán|zpráv"):
        corpus_tables(plans, notes, codes)


def test_conflicting_patient_sexes_are_refused() -> None:
    """Jedna osoba nemůže mít v plánech rozdílné pohlaví závislé na pořadí zpráv."""
    plans = (PLANS[0], PLANS[1].model_copy(update={"sex": Sex.MALE}), PLANS[2])
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    with pytest.raises(InvalidInputError, match=r"90000001.*pohlaví"):
        corpus_tables(plans, NOTES, codes)


def test_discharge_is_not_before_admission() -> None:
    """Zpráva vznikne při přijetí nebo po něm, u všech přijetí fixture F."""
    tables = corpus_tables(
        EDA_PLANS,
        EDA_NOTES,
        load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES),
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
