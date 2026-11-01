"""Ruční orákula přidaných kódů EDA nezávislá na textu a korpusu repozitáře."""

from __future__ import annotations

from copy import deepcopy
from typing import Final

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mentions import Mention, MentionStatus
from mediparse.domain.note_plan import NotePlan
from mediparse.domain.synthetic_coverage import coverage_diagnoses
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES
from tests.synthetic_support import EDA_PLANS

EXPECTED_ADDED_NOTES: Final = 30


@pytest.fixture
def plans() -> tuple[NotePlan, ...]:
    """Sedmnáct zpráv bez zmínky a samostatné skupiny s negací pneumonie.

    Returns:
        Plány v obráceném pořadí pro ověření výběru podle note_id.
    """
    base = EDA_PLANS[0]
    absent = tuple(
        base.model_copy(
            update={
                "note_id": f"90000001-DS-{index:03}",
                "labels": (Diagnosis.DIABETES,),
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
    return tuple(
        NotePlan.model_validate(plan.model_dump())
        for plan in reversed((*absent, *negated, both_negated, *excluded))
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


@pytest.mark.parametrize(("version", "expected"), [("10", "J12"), ("9", "0221")])
def test_pneumonia_selection(
    plans: tuple[NotePlan, ...], version: str, expected: str
) -> None:
    """Jen negované zmínky a absence mají kód, smíšená nejistota a label ne."""
    rows, hadm = _rows(plans, version)
    original = deepcopy((plans, rows, hadm))
    added = coverage_diagnoses(
        plans, rows, hadm, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)
    )
    assert {row["icd_code"] for row in added} == {expected}
    assert len(added) == EXPECTED_ADDED_NOTES
    assert all(row["seq_num"] == "3" for row in added)
    assert (plans, rows, hadm) == original
    assert {row["hadm_id"] for row in added} == {
        hadm[f"90000001-DS-{index:03}"] for index in (*range(1, 16), *range(101, 115))
    } | {hadm["90000001-DS-0002"]}


@pytest.mark.parametrize("group", ["negated", "absent"])
def test_fourteen_eligible_notes_are_refused(
    plans: tuple[NotePlan, ...], group: str
) -> None:
    """Každá ze dvou skupin musí odmítnout nedostatek místo tichého poklesu pokrytí."""
    start = 100 if group == "negated" else 1
    excluded = {f"90000001-DS-{index:03}" for index in range(start + 14, start + 17)}
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
