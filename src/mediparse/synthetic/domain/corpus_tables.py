"""Tabulky EDA s fakty syntetického korpusu, pokrytím a demografií."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.synthetic.domain.note_structure import Sex
from mediparse.synthetic.domain.synthetic_coverage import coverage_diagnoses
from mediparse.synthetic.domain.synthetic_mimic import TIMESTAMP, mimic_core

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.input_note import InputNote
    from mediparse.synthetic.domain.note_plan import NotePlan

_ERAS: Final = {IcdVersion.NINE: "2008 - 2010", IcdVersion.TEN: "2017 - 2019"}
_ANCHOR_AGES: Final = (15, 25, 35, 45, 55, 65, 75, 91)


def corpus_tables(
    plans: Sequence[NotePlan],
    notes: Sequence[InputNote],
    codes: DiagnosisCodes,
) -> dict[str, list[dict[str, str]]]:
    """Doplní řádky discharge a diagnoses_icd z `mimic_core` o pokrytí EDA, demografii a číselník.

    Returns:
        Pět tabulek EDA s korpusovými fakty a doplněným pokrytím.

    Raises:
        InvalidInputError: Plány neodpovídají zprávám, pohlaví pacienta se liší,
            chybí skupina pokrytí nebo kód nepatří do očekávané skupiny.
    """
    _validate_plans_for_notes(plans, notes)
    tables, versions = mimic_core(notes, codes)
    # Kódy bez labelu jsou svědci EDA; smoketest dál používá jen fakta plánu.
    tables["diagnoses_icd"].extend(
        coverage_diagnoses(
            plans,
            tables["diagnoses_icd"],
            {row["note_id"]: row["hadm_id"] for row in tables["discharge"]},
            codes,
        )
    )
    by_subject = {plan.subject_id: plan for plan in plans}
    indices = {subject: index for index, subject in enumerate(sorted(versions))}
    tables["patients"] = [
        {
            "subject_id": str(subject),
            "gender": {Sex.FEMALE: "F", Sex.MALE: "M"}[by_subject[subject].sex],
            "anchor_age": str(_ANCHOR_AGES[indices[subject] % len(_ANCHOR_AGES)]),
            "anchor_year": "2150",
            "anchor_year_group": _ERAS[versions[subject]],
        }
        for subject in sorted(versions)
    ]
    tables["admissions"] = [
        _admission(row, indices[int(row["subject_id"])]) for row in tables["discharge"]
    ]
    tables["d_icd_diagnoses"] = _dictionary(tables["diagnoses_icd"], codes)
    return tables


def _validate_plans_for_notes(
    plans: Sequence[NotePlan],
    notes: Sequence[InputNote],
) -> None:
    by_note = {plan.note_id: plan for plan in plans}
    note_ids = {note.note_id for note in notes}
    if without_plan := note_ids - by_note.keys():
        msg = f"Zprávy bez plánu: {', '.join(sorted(without_plan))}."
        raise InvalidInputError(msg)
    if without_note := by_note.keys() - note_ids:
        msg = f"Plány bez zprávy: {', '.join(sorted(without_note))}."
        raise InvalidInputError(msg)
    sexes: dict[int, Sex] = {}
    for plan in plans:
        sex = sexes.setdefault(plan.subject_id, plan.sex)
        if sex is not plan.sex:
            msg = f"Pacient {plan.subject_id} má v plánech různé pohlaví."
            raise InvalidInputError(msg)


def _admission(row: Mapping[str, str], patient_index: int) -> dict[str, str]:
    index = patient_index % 2
    return {
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "admittime": TIMESTAMP,
        "insurance": ("Medicare", "Private")[index],
        "marital_status": ("MARRIED", "SINGLE")[index],
        "race": ("WHITE", "BLACK/AFRICAN AMERICAN")[index],
        "language": ("English", "Spanish")[index],
    }


def _dictionary(
    rows: Sequence[Mapping[str, str]], codes: DiagnosisCodes
) -> list[dict[str, str]]:
    pairs = sorted({(row["icd_code"], row["icd_version"]) for row in rows})
    result = []
    for code, version in pairs:
        diagnosis = codes.diagnosis_of(
            IcdCode(version=IcdVersion(int(version)), code=code)
        )
        title = diagnosis.value if diagnosis is not None else "filler"
        result.append({
            "icd_code": code,
            "icd_version": version,
            "long_title": f"{title} ICD-{version}",
        })
    return result
