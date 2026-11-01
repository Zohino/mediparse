"""Ruční orákula přidaných kódů EDA nezávislá na textu a korpusu repozitáře."""

from __future__ import annotations

from copy import deepcopy
from typing import Final

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mentions import Mention, MentionStatus
from mediparse.domain.note_plan import NotePlan
from mediparse.domain.synthetic_coverage import coverage_diagnoses
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES
from tests.synthetic_support import EDA_PLANS

CKD_NOTE_INDEX: Final = 17


@pytest.fixture
def plans() -> tuple[NotePlan, ...]:
    """Šestnáct zpráv bez CKD, jedna s CKD a samostatné skupiny pneumonie.

    Returns:
        Plány v obráceném pořadí pro ověření výběru podle note_id.
    """
    base = EDA_PLANS[0]
    absent = tuple(
        base.model_copy(
            update={
                "note_id": f"90000001-DS-{index:03}",
                "labels": (Diagnosis.CKD,)
                if index == CKD_NOTE_INDEX
                else (Diagnosis.DIABETES,),
                "mentions": (),
            }
        )
        for index in range(1, 18)
    )
    negated = tuple(
        base.model_copy(
            update={
                "note_id": f"90000001-DS-{index:03}",
                "labels": (),
                "mentions": (
                    Mention(
                        diagnosis=Diagnosis.PNEUMONIA,
                        status=MentionStatus.NEGATED
                        if index % 2
                        else MentionStatus.FAMILY_HISTORY,
                        sections=("chief_complaint",),
                    ),
                ),
            }
        )
        for index in range(101, 117)
    )
    excluded = (
        negated[0].model_copy(
            update={
                "note_id": "90000001-DS-000",
                "mentions": (
                    *negated[0].mentions,
                    Mention(
                        diagnosis=Diagnosis.PNEUMONIA,
                        status=MentionStatus.UNCERTAIN,
                        sections=("chief_complaint",),
                    ),
                ),
            }
        ),
        absent[0].model_copy(
            update={
                "note_id": "90000001-DS-0001",
                "labels": (Diagnosis.PNEUMONIA,),
            }
        ),
    )
    both_negated = negated[0].model_copy(
        update={
            "note_id": "90000001-DS-0002",
            "mentions": (
                *negated[0].mentions,
                Mention(
                    diagnosis=Diagnosis.PNEUMONIA,
                    status=MentionStatus.FAMILY_HISTORY,
                    sections=("chief_complaint",),
                ),
            ),
        }
    )
    # Zbytek nese CKD: jediných 16 způsobilých zpráv zůstává v absent.
    return tuple(
        NotePlan.model_validate(plan.model_dump())
        for plan in reversed((
            *absent,
            *(
                p.model_copy(update={"labels": (Diagnosis.CKD,)})
                for p in (*negated, both_negated)
            ),
            *(
                p.model_copy(update={"labels": (*p.labels, Diagnosis.CKD)})
                for p in excluded
            ),
        ))
    )


def _rows(
    plans: tuple[NotePlan, ...], version: str
) -> tuple[list[dict[str, str]], dict[str, str]]:
    hadm = {plan.note_id: str(index) for index, plan in enumerate(plans)}
    return [
        {
            "subject_id": str(plan.subject_id),
            "hadm_id": hadm[plan.note_id],
            "seq_num": seq,
            "icd_code": code,
            "icd_version": version,
        }
        for plan in plans
        for seq, code in (
            ("1", "Z0000" if version == "10" else "V700"),
            ("2", "E08" if version == "10" else "249"),
        )
    ], hadm


@pytest.mark.parametrize(("version", "expected"), [("10", "N19"), ("9", "586")])
def test_first_fifteen_ckd_negative_notes(
    plans: tuple[NotePlan, ...], version: str, expected: str
) -> None:
    """Z 16 zpráv bez CKD dostane kód prvních 15, zpráva s CKD jej nedostane."""
    rows, hadm = _rows(plans, version)
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    added = coverage_diagnoses(plans, rows, hadm, codes)
    assert [row["hadm_id"] for row in added if row["icd_code"] == expected] == [
        hadm[f"90000001-DS-{index:03}"] for index in range(1, 16)
    ]
    assert all(row["seq_num"] == "3" for row in added if row["icd_code"] == expected)
    assert len({(row["hadm_id"], row["seq_num"]) for row in (*rows, *added)}) == len(
        rows
    ) + len(added)


def test_pneumonia_selection(plans: tuple[NotePlan, ...]) -> None:
    """Jen negované zmínky a absence mají kód, smíšená nejistota a label ne."""
    rows, hadm = _rows(plans, "10")
    added = coverage_diagnoses(
        plans, rows, hadm, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    )
    assert {row["hadm_id"] for row in added if row["icd_code"] == "J12"} == {
        hadm[f"90000001-DS-{index:03}"] for index in (*range(1, 16), *range(101, 115))
    } | {hadm["90000001-DS-0002"]}


@pytest.mark.parametrize("group", ["ckd", "copd", "negated", "absent"])
def test_fourteen_eligible_notes_are_refused(
    plans: tuple[NotePlan, ...], group: str
) -> None:
    """Každá ze čtyř skupin musí odmítnout nedostatek místo tichého poklesu pokrytí."""
    if group in {"ckd", "copd"}:
        diagnosis = Diagnosis(group)
        eligible = sorted(
            (p for p in plans if diagnosis not in p.labels), key=lambda p: p.note_id
        )
        excluded = {p.note_id for p in eligible[14:]}
        plans = tuple(
            p.model_copy(update={"labels": (*p.labels, diagnosis)})
            if p.note_id in excluded
            else p
            for p in plans
        )
    else:
        start = 100 if group == "negated" else 1
        excluded = {
            f"90000001-DS-{index:03}" for index in range(start + 14, start + 17)
        }
        plans = tuple(
            p.model_copy(
                update={
                    "mentions": (
                        Mention(
                            diagnosis=Diagnosis.PNEUMONIA,
                            status=MentionStatus.UNCERTAIN,
                            sections=("chief_complaint",),
                        ),
                    )
                }
            )
            if p.note_id in excluded
            else p
            for p in plans
        )
    rows, hadm = _rows(plans, "10")
    with pytest.raises(
        InvalidInputError, match=rf"Skupina pokrytí {group} potřebuje 15 zpráv, má 14\."
    ):
        coverage_diagnoses(
            plans, rows, hadm, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
        )


@pytest.mark.parametrize("version", ["9", "10"])
def test_comorbidity_codes_are_outside_candidates_and_inputs_unchanged(
    plans: tuple[NotePlan, ...], version: str
) -> None:
    """Kódy Charlsonu nerozšiřují kandidáty a čistá funkce zachová své vstupy."""
    rows, hadm = _rows(plans, version)
    original = deepcopy((plans, rows, hadm))
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    added = coverage_diagnoses(plans, rows, hadm, codes)
    comorbidity = [
        row
        for row in added
        if row["icd_code"] != ("J12" if version == "10" else "0221")
    ]
    assert comorbidity
    assert all(
        codes.diagnosis_of(
            IcdCode(version=IcdVersion(int(version)), code=row["icd_code"])
        )
        is None
        for row in comorbidity
    )
    assert (plans, rows, hadm) == original


def test_comorbidity_code_belonging_to_candidate_is_refused(
    plans: tuple[NotePlan, ...],
) -> None:
    """Širší předpona kandidáta nesmí pohltit kód pokrytí Charlsonovy kategorie."""
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    prefixes = dict(codes.prefixes)
    prefixes[Diagnosis.CKD] = {**prefixes[Diagnosis.CKD], IcdVersion.TEN: ("N1",)}
    overlapping = DiagnosisCodes(prefixes=prefixes)
    rows, hadm = _rows(plans, "10")
    with pytest.raises(InvalidInputError, match=r"N19 ICD-10 patří kandidátovi"):
        coverage_diagnoses(plans, rows, hadm, overlapping)


def test_missing_pneumonia_prefix_for_admission_version_is_refused(
    plans: tuple[NotePlan, ...],
) -> None:
    """Hospitalizace ICD-9 potřebuje kód pneumonie právě v této verzi."""
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    prefixes = dict(codes.prefixes)
    prefixes[Diagnosis.PNEUMONIA] = {
        IcdVersion.TEN: prefixes[Diagnosis.PNEUMONIA][IcdVersion.TEN]
    }
    without_nine = DiagnosisCodes(prefixes=prefixes)
    rows, hadm = _rows(plans, "9")
    with pytest.raises(InvalidInputError, match=r"pneumonia nemá předponu ICD-9"):
        coverage_diagnoses(plans, rows, hadm, without_nine)
