"""Řádky vstupních tabulek MIMIC z párovaných syntetických zpráv a jejich labelů."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Final

from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note import order_of

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
_TIMESTAMP: Final = "2150-01-01 00:00:00"


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
    for version, filler in FILLER_CODES.items():
        if codes.diagnosis_of(IcdCode(version=version, code=filler)) is not None:
            msg = f"Výplňový kód ICD-{version.value} patří do diagnózy pětice."
            raise InvalidInputError(msg)
    tables: dict[str, list[dict[str, str]]] = {"discharge": [], "diagnoses_icd": []}
    for index, note in enumerate(sorted(notes, key=lambda note: note.note_id)):
        hadm = str(_FIRST_ADMISSION + index)
        # Střídání verzí procvičí obě mapování předpon ve skutečné kohortě.
        version = IcdVersion.TEN if index % 2 == 0 else IcdVersion.NINE
        tables["discharge"].append(_discharge(note, hadm))
        tables["diagnoses_icd"].extend(_diagnoses(note, hadm, version, codes))
    _validate_shapes(shapes, tables)
    return tables


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
