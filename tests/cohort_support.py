"""Zkonstruované tabulky MIMIC pro testy kohorty: řetězcové sloupce jako v převedeném parquetu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.parquet_file import write_string_table

if TYPE_CHECKING:
    from pathlib import Path

CODES: Final = DiagnosisCodes(
    prefixes={
        Diagnosis.DIABETES: {IcdVersion.NINE: ("250",), IcdVersion.TEN: ("E11",)},
        Diagnosis.COPD: {IcdVersion.NINE: ("496",), IcdVersion.TEN: ("J44",)},
        Diagnosis.CKD: {IcdVersion.NINE: ("585",), IcdVersion.TEN: ("N18",)},
        Diagnosis.PNEUMONIA: {IcdVersion.NINE: ("486",), IcdVersion.TEN: ("J18",)},
        Diagnosis.ATRIAL_FIBRILLATION: {
            IcdVersion.NINE: ("4273",),
            IcdVersion.TEN: ("I48",),
        },
    }
)
PADDING_CODES: Final = {
    Diagnosis.DIABETES: ("10", "E119"),
    Diagnosis.COPD: ("10", "J449"),
    Diagnosis.CKD: ("9", "5859"),
    Diagnosis.PNEUMONIA: ("10", "J189"),
    Diagnosis.ATRIAL_FIBRILLATION: ("10", "I4891"),
}
PADDING_PER_DIAGNOSIS: Final = 10
DISCHARGE_COLUMNS: Final = ("note_id", "subject_id", "hadm_id", "text")
DIAGNOSES_COLUMNS: Final = (
    "subject_id",
    "hadm_id",
    "seq_num",
    "icd_code",
    "icd_version",
)

type NoteRow = dict[str, str | None]
type IcdRow = dict[str, str | None]


def text(tag: str) -> str:
    """Jedinečný text o více než sto znacích.

    Returns:
        Text, který projde filtrem délky.
    """
    return f"{tag} " + "word " * 30


def note(note_id: str, subject: int, hadm: int | None, body: str | None) -> NoteRow:
    """Řádek ``discharge``.

    Returns:
        Řádek se samými řetězci a NULL tam, kde chybí hodnota.
    """
    return {
        "note_id": note_id,
        "subject_id": str(subject),
        "hadm_id": None if hadm is None else str(hadm),
        "text": body,
    }


def icd(hadm: int, seq: int, code: str | None, version: str | None) -> IcdRow:
    """Řádek ``diagnoses_icd``.

    Returns:
        Řádek se samými řetězci.
    """
    return {
        "subject_id": "0",
        "hadm_id": str(hadm),
        "seq_num": str(seq),
        "icd_code": code,
        "icd_version": version,
    }


def base_notes() -> list[NoteRow]:
    """Dvanáct zpráv, které procvičí každý filtr trychtýře.

    Returns:
        Řádky ``discharge``.
    """
    return [
        note("n01", 1, 10, text("a")),
        note("n02", 2, 20, text("b")),
        note("n03", 3, None, text("f")),
        note("n04", 4, 40, text("g")),
        note("n05", 5, 50, "short"),
        note("n06", 6, 60, None),
        note("n07", 7, 70, text("h")),
        note("n08", 1, 80, text("a")),
        note("n09", 8, 90, text("c")),
        note("n10", 9, 100, text("c")),
        note("n11", 10, 110, text("d")),
        note("n12", 10, 110, text("e")),
    ]


def base_icd() -> list[IcdRow]:
    """Řádky ICD k základním zprávám.

    Returns:
        Řádky ``diagnoses_icd``.
    """
    return [
        icd(10, 1, "I10", "10"),
        icd(10, 2, "E119", "10"),
        icd(20, 3, "4273", "9"),
        icd(50, 1, "E119", "10"),
        icd(60, 1, "E119", "10"),
        icd(70, 1, None, "10"),
        icd(80, 1, "N189", "10"),
        icd(100, 1, "J189", "10"),
        icd(110, 1, "J449", "10"),
    ]


def padding() -> tuple[list[NoteRow], list[IcdRow]]:
    """Deset zpráv na diagnózu, aby žádná buňka nebyla pod prahem.

    Returns:
        Řádky ``discharge`` a ``diagnoses_icd``.
    """
    notes: list[NoteRow] = []
    codes: list[IcdRow] = []
    for index, diagnosis in enumerate(Diagnosis):
        version, code = PADDING_CODES[diagnosis]
        for k in range(PADDING_PER_DIAGNOSIS):
            hadm = 1000 + index * 100 + k
            notes.append(note(f"p{index}-{k:02d}", hadm, hadm, text(f"p{hadm}")))
            codes.append(icd(hadm, 1, code, version))
    return notes, codes


def write_tables(
    root: Path,
    notes: list[NoteRow] | None = None,
    codes: list[IcdRow] | None = None,
    *,
    padded: bool = True,
) -> tuple[Path, Path]:
    """Zapíše ``discharge.parquet`` a ``diagnoses_icd.parquet``.

    Returns:
        Cesty k oběma souborům.
    """
    extra_notes, extra_codes = padding() if padded else ([], [])
    discharge = root / "discharge.parquet"
    diagnoses = root / "diagnoses_icd.parquet"
    write_string_table(
        discharge, DISCHARGE_COLUMNS, [*(notes or base_notes()), *extra_notes]
    )
    write_string_table(
        diagnoses, DIAGNOSES_COLUMNS, [*(codes or base_icd()), *extra_codes]
    )
    return discharge, diagnoses
