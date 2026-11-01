"""Řádky vstupních tabulek MIMIC z párovaných syntetických zpráv a jejich labelů."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Final

from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note import order_of
from mediparse.domain.note_structure import Sex
from mediparse.domain.synthetic_coverage import coverage_diagnoses

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.language import Language
    from mediparse.domain.mimic_table import TableShape
    from mediparse.domain.note_plan import NotePlan

# Bez kódu ICD by filtr has_icd vyřadil zprávy bez labelu pětice.
FILLER_CODES: Final[Mapping[IcdVersion, str]] = {
    IcdVersion.NINE: "V700",
    IcdVersion.TEN: "Z0000",
}
_FIRST_ADMISSION: Final = 20000000
_TIMESTAMP: Final = "2155-01-01 00:00:00"
_ERAS: Final = {IcdVersion.NINE: "2008 - 2010", IcdVersion.TEN: "2017 - 2019"}
_ANCHOR_AGES: Final = (15, 25, 35, 45, 55, 65, 75, 91)


class UnpairedNotesError(InvalidInputError):
    """Plány a zprávy jednoho jazyka si neodpovídají jedna k jedné."""


def input_notes(
    plans: Sequence[NotePlan], texts: Mapping[str, str], language: Language
) -> tuple[InputNote, ...]:
    """Spáruje plány se zprávami jednoho jazyka.

    Args:
        plans: Plány zpráv korpusu.
        texts: Texty zpráv klíčované note_id.
        language: Jazyk zpráv.

    Returns:
        Řádky seřazené podle note_id.

    Raises:
        UnpairedNotesError: Korpus je prázdný, plán je duplicitní nebo plán
            a zpráva nemají protějšek.
    """
    counts = Counter(plan.note_id for plan in plans)
    if problems := _pairing_problems(counts, texts, language):
        raise UnpairedNotesError(" ".join(problems))
    by_id = {plan.note_id: plan for plan in plans}
    return tuple(
        InputNote(
            note_id,
            by_id[note_id].subject_id,
            language,
            texts[note_id],
            frozenset(by_id[note_id].labels),
        )
        for note_id in sorted(by_id)
    )


def _pairing_problems(
    counts: Counter[str], texts: Mapping[str, str], language: Language
) -> list[str]:
    if not counts and not texts:
        return [f"Korpus nemá v jazyce {language} žádnou zprávu ani plán."]
    problems = []
    if duplicates := sorted(note_id for note_id, count in counts.items() if count > 1):
        problems.append(f"Duplicitní plány: {', '.join(duplicates)}.")
    if without_text := sorted(counts.keys() - texts.keys()):
        problems.append(
            f"Plány bez zprávy v jazyce {language}: {', '.join(without_text)}."
        )
    if without_plan := sorted(texts.keys() - counts.keys()):
        problems.append(
            f"Zprávy bez plánu v jazyce {language}: {', '.join(without_plan)}."
        )
    return problems


def synthetic_mimic_rows(
    notes: Sequence[InputNote],
    codes: DiagnosisCodes,
    shapes: Mapping[str, TableShape],
) -> dict[str, list[dict[str, str]]]:
    """Převede labely zpráv na kódy ICD a ověří sloupce požadované manifestem.

    Returns:
        Tabulky discharge a diagnoses_icd se samými řetězci.

    Raises:
        InvalidInputError: Chybí tvar tabulky, sloupec není známý, výplň patří
            do pětice nebo diagnóza nemá předponu v použité verzi ICD.
    """
    tables, _ = _mimic_core(notes, codes)
    _validate_shapes(shapes, tables)
    return tables


def _mimic_core(
    notes: Sequence[InputNote],
    codes: DiagnosisCodes,
) -> tuple[dict[str, list[dict[str, str]]], dict[int, IcdVersion]]:
    for version, filler in FILLER_CODES.items():
        if codes.diagnosis_of(IcdCode(version=version, code=filler)) is not None:
            msg = f"Výplňový kód ICD-{version.value} patří do diagnózy pětice."
            raise InvalidInputError(msg)
    tables: dict[str, list[dict[str, str]]] = {"discharge": [], "diagnoses_icd": []}
    versions = _patient_versions(notes)
    for index, note in enumerate(sorted(notes, key=lambda note: note.note_id)):
        hadm = str(_FIRST_ADMISSION + index)
        version = versions[note.subject_id]
        tables["discharge"].append(_discharge(note, hadm))
        tables["diagnoses_icd"].extend(_diagnoses(note, hadm, version, codes))
    return tables, versions


def _patient_versions(notes: Sequence[InputNote]) -> dict[int, IcdVersion]:
    # Období pacienta i všechny jeho hospitalizace musí sdílet jednu verzi ICD.
    return {
        subject: IcdVersion.TEN if index % 2 == 0 else IcdVersion.NINE
        for index, subject in enumerate(sorted({note.subject_id for note in notes}))
    }


def _validate_shapes(
    shapes: Mapping[str, TableShape], tables: Mapping[str, list[dict[str, str]]]
) -> None:
    for table, rows in tables.items():
        if table not in shapes:
            msg = f"V manifestu chybí tabulka {table}."
            raise InvalidInputError(msg)
        columns = set(shapes[table].columns)
        for row in rows:
            if unknown := columns - row.keys():
                msg = (
                    f"Tabulka {table} má neznámé sloupce: {', '.join(sorted(unknown))}."
                )
                raise InvalidInputError(msg)


def _discharge(note: InputNote, hadm: str) -> dict[str, str]:
    return {
        "note_id": note.note_id,
        "subject_id": str(note.subject_id),
        "hadm_id": hadm,
        "note_type": "DS",
        "note_seq": str(order_of(note.note_id)),
        "charttime": _TIMESTAMP,
        "storetime": _TIMESTAMP,
        "text": note.text,
    }


def _diagnoses(
    note: InputNote, hadm: str, version: IcdVersion, codes: DiagnosisCodes
) -> list[dict[str, str]]:
    icd_codes = [FILLER_CODES[version]]
    # Pořadí frozensetu závisí na PYTHONHASHSEED; seřazení drží seq_num a otisky parquetů.
    for diagnosis in sorted(note.labels):
        code = codes.representative(diagnosis, version)
        if code is None:
            msg = f"Diagnóza {diagnosis.value} nemá předponu ICD-{version.value}."
            raise InvalidInputError(msg)
        icd_codes.append(code)
    return [
        {
            "subject_id": str(note.subject_id),
            "hadm_id": hadm,
            "seq_num": str(seq),
            "icd_code": code,
            "icd_version": str(version.value),
        }
        for seq, code in enumerate(icd_codes, start=1)
    ]


def corpus_tables(
    plans: Sequence[NotePlan],
    notes: Sequence[InputNote],
    codes: DiagnosisCodes,
) -> dict[str, list[dict[str, str]]]:
    """Doplní jádro smoketestu o pokrytí EDA, demografii a číselník.

    Returns:
        Pět tabulek EDA s korpusovými fakty a doplněným pokrytím.

    Raises:
        InvalidInputError: Plány neodpovídají zprávám, pohlaví pacienta se liší,
            chybí skupina pokrytí nebo kód nepatří do očekávané skupiny.
    """
    _validate_plans_for_notes(plans, notes)
    tables, versions = _mimic_core(notes, codes)
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
        "admittime": _TIMESTAMP,
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
