"""Kódy navíc pro pokrytí větví EDA nad zmrazeným syntetickým korpusem."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mentions import MentionStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.note_plan import NotePlan

# Práh zveřejnění 10 + rezerva 5 pro rozdíl mezi plánem a detekcí zmínek v R.
COMORBIDITY_NOTES: Final = 15
MENTION_NOTES: Final = 15
# Mapy Quan v R comorbidity (tidy.codes = TRUE): rend a cpd jsou širší než
# kandidáti CKD a CHOPN. Tyto kódy mimo předpony kandidátů odkrývají jejich rozdíl.
_COMORBIDITY_WITNESSES: Final = {
    Diagnosis.CKD: {IcdVersion.TEN: "N19", IcdVersion.NINE: "586"},
    Diagnosis.COPD: {IcdVersion.TEN: "J45", IcdVersion.NINE: "4930"},
}
_NEGATED: Final = frozenset({MentionStatus.NEGATED, MentionStatus.FAMILY_HISTORY})


def coverage_diagnoses(
    plans: Sequence[NotePlan],
    diagnoses: Sequence[Mapping[str, str]],
    hadm_of: Mapping[str, str],
    codes: DiagnosisCodes,
) -> list[dict[str, str]]:
    """Vybere zprávy podle plánů a vrátí jen nové řádky diagnóz pro EDA.

    Args:
        plans: Plány korpusu v libovolném pořadí.
        diagnoses: Původní diagnózy včetně výplně s verzí ICD hospitalizace.
        hadm_of: Hospitalizace každé zprávy.
        codes: Předpony kandidátů pětice.

    Returns:
        Přidané kódy s navazujícím seq_num, bez změny vstupních řádků a labelů.

    Raises:
        InvalidInputError: Skupina nemá dost zpráv, kód Charlsonu patří kandidátovi
            nebo pneumonie nemá reprezentativní kód v použité verzi.
    """
    ordered = sorted(plans, key=lambda plan: plan.note_id)
    selected: list[tuple[NotePlan, Diagnosis]] = []
    for diagnosis in _COMORBIDITY_WITNESSES:
        eligible = [plan for plan in ordered if diagnosis not in plan.labels]
        selected.extend(
            (plan, diagnosis)
            for plan in _first(eligible, COMORBIDITY_NOTES, diagnosis.value)
        )
    pneumonia = [plan for plan in ordered if Diagnosis.PNEUMONIA not in plan.labels]
    for group, eligible in (
        (
            "negated",
            [
                plan
                for plan in pneumonia
                if (statuses := _pneumonia_statuses(plan)) and statuses <= _NEGATED
            ],
        ),
        ("absent", [plan for plan in pneumonia if not _pneumonia_statuses(plan)]),
    ):
        selected.extend(
            (plan, Diagnosis.PNEUMONIA)
            for plan in _first(eligible, MENTION_NOTES, group)
        )
    return _diagnosis_rows(selected, diagnoses, hadm_of, codes)


def _first(plans: Sequence[NotePlan], count: int, group: str) -> Sequence[NotePlan]:
    if len(plans) < count:
        msg = f"Skupina pokrytí {group} potřebuje {count} zpráv, má {len(plans)}."
        raise InvalidInputError(msg)
    return plans[:count]


def _pneumonia_statuses(plan: NotePlan) -> frozenset[MentionStatus]:
    return frozenset(
        mention.status
        for mention in plan.mentions
        if mention.diagnosis is Diagnosis.PNEUMONIA
    )


def _diagnosis_rows(
    selected: Sequence[tuple[NotePlan, Diagnosis]],
    diagnoses: Sequence[Mapping[str, str]],
    hadm_of: Mapping[str, str],
    codes: DiagnosisCodes,
) -> list[dict[str, str]]:
    versions = {
        row["hadm_id"]: IcdVersion(int(row["icd_version"])) for row in diagnoses
    }
    sequences: dict[str, int] = defaultdict(int)
    for row in diagnoses:
        sequences[row["hadm_id"]] = max(sequences[row["hadm_id"]], int(row["seq_num"]))
    result = []
    for plan, diagnosis in selected:
        hadm = hadm_of[plan.note_id]
        version = versions[hadm]
        code = _coverage_code(diagnosis, version, codes)
        sequences[hadm] += 1
        result.append({
            "subject_id": str(plan.subject_id),
            "hadm_id": hadm,
            "seq_num": str(sequences[hadm]),
            "icd_code": code,
            "icd_version": str(version.value),
        })
    return result


def _coverage_code(
    diagnosis: Diagnosis, version: IcdVersion, codes: DiagnosisCodes
) -> str:
    if diagnosis in _COMORBIDITY_WITNESSES:
        code = _COMORBIDITY_WITNESSES[diagnosis][version]
        if codes.diagnosis_of(IcdCode(version=version, code=code)) is not None:
            msg = f"Kód pokrytí {code} ICD-{version.value} patří kandidátovi pětice."
            raise InvalidInputError(msg)
        return code
    code = codes.representative(diagnosis, version)
    if code is None:
        msg = f"Diagnóza {diagnosis.value} nemá předponu ICD-{version.value}."
        raise InvalidInputError(msg)
    return code
