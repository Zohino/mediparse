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
    versions = {
        row["hadm_id"]: IcdVersion(int(row["icd_version"])) for row in diagnoses
    }
    selected: list[tuple[NotePlan, IcdCode]] = []
    for diagnosis, witnesses in _COMORBIDITY_WITNESSES.items():
        eligible = [plan for plan in ordered if diagnosis not in plan.labels]
        for plan in _first(eligible, COMORBIDITY_NOTES, diagnosis.value):
            version = versions[hadm_of[plan.note_id]]
            selected.append((
                plan,
                _comorbidity_code(witnesses[version], version, codes),
            ))
    pneumonia = [plan for plan in ordered if Diagnosis.PNEUMONIA not in plan.labels]
    for group, predicate in (("negated", _only_negated), ("absent", _unmentioned)):
        eligible = [plan for plan in pneumonia if predicate(plan)]
        for plan in _first(eligible, MENTION_NOTES, group):
            version = versions[hadm_of[plan.note_id]]
            selected.append((plan, _pneumonia_code(version, codes)))
    return _diagnosis_rows(selected, diagnoses, hadm_of)


def _first(plans: Sequence[NotePlan], count: int, group: str) -> Sequence[NotePlan]:
    if len(plans) < count:
        msg = f"Skupina pokrytí {group} potřebuje {count} zpráv, má {len(plans)}."
        raise InvalidInputError(msg)
    return plans[:count]


def _only_negated(plan: NotePlan) -> bool:
    statuses = _pneumonia_statuses(plan)
    return bool(statuses) and statuses <= _NEGATED


def _unmentioned(plan: NotePlan) -> bool:
    return not _pneumonia_statuses(plan)


def _pneumonia_statuses(plan: NotePlan) -> frozenset[MentionStatus]:
    return frozenset(
        mention.status
        for mention in plan.mentions
        if mention.diagnosis is Diagnosis.PNEUMONIA
    )


def _diagnosis_rows(
    selected: Sequence[tuple[NotePlan, IcdCode]],
    diagnoses: Sequence[Mapping[str, str]],
    hadm_of: Mapping[str, str],
) -> list[dict[str, str]]:
    sequences: dict[str, int] = defaultdict(int)
    for row in diagnoses:
        sequences[row["hadm_id"]] = max(sequences[row["hadm_id"]], int(row["seq_num"]))
    result = []
    for plan, code in selected:
        hadm = hadm_of[plan.note_id]
        sequences[hadm] += 1
        result.append({
            "subject_id": str(plan.subject_id),
            "hadm_id": hadm,
            "seq_num": str(sequences[hadm]),
            "icd_code": code.code,
            "icd_version": str(code.version.value),
        })
    return result


def _comorbidity_code(code: str, version: IcdVersion, codes: DiagnosisCodes) -> IcdCode:
    icd = IcdCode(version=version, code=code)
    if codes.diagnosis_of(icd) is not None:
        msg = f"Kód pokrytí {code} ICD-{version.value} patří kandidátovi pětice."
        raise InvalidInputError(msg)
    return icd


def _pneumonia_code(version: IcdVersion, codes: DiagnosisCodes) -> IcdCode:
    code = codes.representative(Diagnosis.PNEUMONIA, version)
    if code is None:
        msg = f"Diagnóza pneumonia nemá předponu ICD-{version.value}."
        raise InvalidInputError(msg)
    return IcdCode(version=version, code=code)
