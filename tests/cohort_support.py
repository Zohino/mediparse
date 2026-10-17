"""Zkonstruované tabulky MIMIC pro testy kohorty: řetězcové sloupce jako v převedeném parquetu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mediparse.domain.cohort import CohortCounts, FunnelStep, FunnelStepName
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.infrastructure.parquet_file import write_string_table

if TYPE_CHECKING:
    from pathlib import Path

PREFIXES: Final[dict[Diagnosis, dict[IcdVersion, tuple[str, ...]]]] = {
    Diagnosis.DIABETES: {IcdVersion.NINE: ("250",), IcdVersion.TEN: ("E11",)},
    Diagnosis.COPD: {IcdVersion.NINE: ("496",), IcdVersion.TEN: ("J44",)},
    Diagnosis.CKD: {IcdVersion.NINE: ("585",), IcdVersion.TEN: ("N18",)},
    Diagnosis.PNEUMONIA: {IcdVersion.NINE: ("486",), IcdVersion.TEN: ("J18",)},
    Diagnosis.ATRIAL_FIBRILLATION: {
        IcdVersion.NINE: ("4273",),
        IcdVersion.TEN: ("I48",),
    },
}
CODES: Final = DiagnosisCodes(prefixes=PREFIXES)


def candidate_prefixes() -> dict[str, dict[str, list[str]]]:
    """Předpony pětice ve tvaru kandidátů v ``diagnosis_candidates.json``.

    Returns:
        Pro každou diagnózu seznamy předpon ``icd9_prefixes`` a ``icd10_prefixes``.
    """
    return {
        diagnosis.value: {
            "icd9_prefixes": list(by_version[IcdVersion.NINE]),
            "icd10_prefixes": list(by_version[IcdVersion.TEN]),
        }
        for diagnosis, by_version in PREFIXES.items()
    }


PADDING_CODES: Final = {
    Diagnosis.DIABETES: ("10", "E119"),
    Diagnosis.COPD: ("10", "J449"),
    Diagnosis.CKD: ("9", "5859"),
    Diagnosis.PNEUMONIA: ("10", "J189"),
    Diagnosis.ATRIAL_FIBRILLATION: ("10", "I4891"),
}
PADDING_PER_DIAGNOSIS: Final = 10
PADDING_NOTES: Final = len(Diagnosis) * PADDING_PER_DIAGNOSIS
DISCHARGE_COLUMNS: Final = ("note_id", "subject_id", "hadm_id", "text")
DIAGNOSES_COLUMNS: Final = (
    "subject_id",
    "hadm_id",
    "seq_num",
    "icd_code",
    "icd_version",
)

UNIQUE: Final = 50


def make_counts(
    positives: dict[Diagnosis, int] | None = None,
    unique: int = UNIQUE,
    funnel: tuple[FunnelStep, ...] | None = None,
) -> CohortCounts:
    """Agregáty kohorty; bez argumentů jsou publikovatelné (polovina zpráv pozitivní).

    Returns:
        Agregáty s trychtýřem o stejném počtu zpráv a pacientů v každém kroku.
    """
    return CohortCounts(
        funnel=funnel
        or tuple(
            FunnelStep(step=step, notes=unique, patients=unique)
            for step in FunnelStepName
        ),
        positives=positives or dict.fromkeys(Diagnosis, unique // 2),
    )


FEW_POSITIVES: Final = make_counts({
    **dict.fromkeys(Diagnosis, UNIQUE // 2),
    Diagnosis.COPD: 7,
})
FEW_NEGATIVES: Final = make_counts({
    **dict.fromkeys(Diagnosis, UNIQUE // 2),
    Diagnosis.DIABETES: 43,
})
EMPTY_COHORT: Final = make_counts(dict.fromkeys(Diagnosis, 0), unique=0)

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

    Filtry (pořadí jako ``FunnelStepName``) vyřadí tyto zprávy:

    - ``WITH_ICD``: n03 (bez ``hadm_id``), n04 a n09 (přijetí bez řádku ICD);
    - ``LONG_ENOUGH``: n05 (krátký text), n06 (text NULL);
    - ``UNIQUE``: n08, kopie textu n01 s vyšším ``note_id``.

    n07 má jediný řádek ICD s NULL kódem, takže filtr ``WITH_ICD`` projde bez
    štítků. Pacient 1 má n01 i n08, pacient 10 dvě zprávy téhož přijetí.

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


# Příspěvek zpráv ``base_notes()`` k trychtýři (zprávy, pacienti), odvozený z
# ``base_icd()``. Každý krok odečte zprávy, které vypadly filtrem; viz docstring
# ``base_notes()``. Padding přidá do každého kroku ``PADDING_NOTES`` zpráv od
# tolika pacientů (jeden pacient na zprávu), všechny projdou všemi filtry.
_BASE_FUNNEL: Final = {
    FunnelStepName.ALL_NOTES: (12, 10),
    FunnelStepName.WITH_ICD: (9, 7),  # bez n03, n04, n09
    FunnelStepName.LONG_ENOUGH: (7, 5),  # bez n05, n06
    FunnelStepName.UNIQUE: (6, 5),  # bez n08, kopie textu n01
}
EXPECTED_FUNNEL: Final = {
    step: (notes + PADDING_NOTES, patients + PADDING_NOTES)
    for step, (notes, patients) in _BASE_FUNNEL.items()
}
# Pozitivní zprávy ``base_notes()`` po deduplikaci: n01 cukrovka, n02 fibrilace,
# n10 zápal plic, n11 a n12 (jedno přijetí) CHOPN. Padding dá každé diagnóze
# ``PADDING_PER_DIAGNOSIS`` pozitivních.
_BASE_POSITIVES: Final = {
    Diagnosis.DIABETES: 1,
    Diagnosis.COPD: 2,
    Diagnosis.CKD: 0,
    Diagnosis.PNEUMONIA: 1,
    Diagnosis.ATRIAL_FIBRILLATION: 1,
}
EXPECTED_POSITIVES: Final = {
    diagnosis: base + PADDING_PER_DIAGNOSIS
    for diagnosis, base in _BASE_POSITIVES.items()
}


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
