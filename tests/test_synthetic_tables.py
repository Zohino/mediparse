"""Syntetické řádky tabulky ve tvaru MIMIC: jedinečné buňky, v nichž je vidět prefix."""

from __future__ import annotations

import re
from typing import Final

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.dua import MIN_CELL
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.synthetic_tables import (
    DEFAULT_ROWS,
    RARE_ROWS,
    synthetic_rows,
)
from tests.cohort_support import CODES, PREFIXES

ROWS: Final = 12
TEXT_CHARS: Final = 100
SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "text"))
NOTE_SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "hadm_id", "text"))
PREFIX: Final = "eda-sentinel"
COLUMNS: Final = {
    "patients": (
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
        "dod",
    ),
    "admissions": (
        "hadm_id",
        "admittime",
        "admission_type",
        "insurance",
        "marital_status",
        "race",
        "language",
    ),
    "diagnoses_icd": ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
    "d_icd_diagnoses": ("icd_code", "icd_version", "long_title"),
}
BLOCK_DIAGNOSES: Final = (Diagnosis.DIABETES, Diagnosis.COPD)
BLOCK_VERSIONS: Final = (
    IcdVersion.NINE,
    IcdVersion.NINE,
    IcdVersion.TEN,
    IcdVersion.TEN,
)
BLOCK_CODES: Final = tuple(
    PREFIXES[BLOCK_DIAGNOSES[block % 2]][BLOCK_VERSIONS[block]][0] for block in range(4)
)
# sloupec -> hodnoty bloků 0-3 (slovník MIMIC)
EXPECTED: Final = {
    ("patients", "gender"): ("F", "M", "F", "M"),
    ("patients", "anchor_year_group"): ("2008 - 2010",) * 2 + ("2017 - 2019",) * 2,
    ("admissions", "insurance"): ("Medicare", "Private", "Medicare", "Private"),
    ("admissions", "marital_status"): ("MARRIED", "SINGLE", "MARRIED", "SINGLE"),
    ("admissions", "race"): ("WHITE", "BLACK/AFRICAN AMERICAN") * 2,
    ("admissions", "language"): ("English", "Spanish", "English", "Spanish"),
    ("diagnoses_icd", "icd_version"): ("9", "9", "10", "10"),
    ("diagnoses_icd", "icd_code"): BLOCK_CODES,
}
WITNESSES: Final = {
    ("patients", "anchor_year_group"),
    ("diagnoses_icd", "icd_version"),
    ("diagnoses_icd", "icd_code"),
}
TYPED_TABLES: Final = ("patients", "admissions", "diagnoses_icd")


def _table(table: str, rows: int = DEFAULT_ROWS) -> list[dict[str, str]]:
    shape = TableShape(records=rows, columns=COLUMNS[table])
    return synthetic_rows(
        table, shape, PREFIX, rows, text_chars=TEXT_CHARS, codes=CODES
    )


def _column(table: str, column: str, rows: int = DEFAULT_ROWS) -> list[str]:
    return [row[column] for row in _table(table, rows)]


def test_rows_have_requested_count_and_columns_in_manifest_order() -> None:
    """Každý řádek má všechny sloupce v pořadí manifestu a řádků je tolik, kolik se žádá."""
    rows = synthetic_rows(
        "discharge", SHAPE, "sentinel", ROWS, text_chars=TEXT_CHARS, codes=CODES
    )

    assert len(rows) == ROWS
    assert all(tuple(row) == SHAPE.columns for row in rows)


def test_cells_are_unique_within_table_and_carry_prefix() -> None:
    """Každá buňka tabulky je v ní jedinečná a obsahuje prefix, takže únik zachytí hledání prefixu."""
    rows = synthetic_rows(
        "discharge", SHAPE, "sentinel", ROWS, text_chars=TEXT_CHARS, codes=CODES
    )
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all("sentinel" in cell for cell in cells)


def test_cells_name_table_column_and_row() -> None:
    """Buňka je ve tvaru prefix-tabulka-sloupec-řádek."""
    rows = synthetic_rows(
        "radiology", NOTE_SHAPE, "sentinel", 2, text_chars=TEXT_CHARS, codes=CODES
    )

    assert rows[1]["note_id"] == "sentinel-radiology-note_id-1"


def test_rows_are_deterministic() -> None:
    """Stejné vstupy dávají stejné řádky."""
    assert synthetic_rows(
        "t", SHAPE, "p", 3, text_chars=TEXT_CHARS, codes=CODES
    ) == synthetic_rows("t", SHAPE, "p", 3, text_chars=TEXT_CHARS, codes=CODES)


def test_zero_rows_are_refused() -> None:
    """Nulový počet řádků je chyba."""
    with pytest.raises(InvalidInputError, match="řádků"):
        synthetic_rows("t", SHAPE, "p", 0, text_chars=TEXT_CHARS, codes=CODES)


def test_table_without_columns_is_refused() -> None:
    """Tabulka bez sloupců je chyba a jmenuje tabulku."""
    with pytest.raises(InvalidInputError, match="empty"):
        synthetic_rows(
            "empty",
            TableShape(records=0, columns=()),
            "p",
            3,
            text_chars=TEXT_CHARS,
            codes=CODES,
        )


def test_empty_prefix_is_refused() -> None:
    """Prázdný prefix je chyba, protože kanárek by nic nehledal."""
    with pytest.raises(InvalidInputError, match="Prefix"):
        synthetic_rows("t", SHAPE, "", ROWS, text_chars=TEXT_CHARS, codes=CODES)


def test_patient_and_admission_keys_are_shared_across_tables() -> None:
    """Řádek r každé tabulky patří pacientovi a hospitalizaci r, takže se tabulky spojí jako v MIMIC."""
    notes = synthetic_rows(
        "discharge", NOTE_SHAPE, "p", ROWS, text_chars=TEXT_CHARS, codes=CODES
    )
    codes = synthetic_rows(
        "diagnoses_icd",
        TableShape(records=5, columns=("subject_id", "hadm_id", "icd_code")),
        "p",
        ROWS,
        text_chars=TEXT_CHARS,
        codes=CODES,
    )

    assert [row["hadm_id"] for row in notes] == [row["hadm_id"] for row in codes]
    assert [row["subject_id"] for row in notes] == [row["subject_id"] for row in codes]
    assert notes[3]["hadm_id"] == "p-admissions-hadm_id-3"
    assert notes[3]["subject_id"] == "p-patients-subject_id-3"


def test_bhc_note_id_refers_to_discharge_and_radiology_keeps_its_own() -> None:
    """BHC se páruje s propouštěcí zprávou přes note_id, radiologie má note_id vlastní jako v MIMIC."""
    shape = TableShape(records=5, columns=("note_id", "input"))

    bhc = synthetic_rows(
        "mimic-iv-bhc", shape, "p", 2, text_chars=TEXT_CHARS, codes=CODES
    )
    radiology = synthetic_rows(
        "radiology", NOTE_SHAPE, "p", 2, text_chars=TEXT_CHARS, codes=CODES
    )

    assert bhc[1]["note_id"] == "p-discharge-note_id-1"
    assert radiology[1]["note_id"] == "p-radiology-note_id-1"


def test_text_starts_with_its_cell_and_reaches_minimum_length() -> None:
    """Text začíná svou buňkou a má aspoň požadovaný počet znaků, aby prošel filtrem délky."""
    rows = synthetic_rows(
        "discharge", NOTE_SHAPE, "p", ROWS, text_chars=TEXT_CHARS, codes=CODES
    )

    assert all(
        row["text"].startswith(f"p-discharge-text-{index} ")
        for index, row in enumerate(rows)
    )
    assert all(len(row["text"]) >= TEXT_CHARS for row in rows)


def test_text_token_is_letters_only_and_unique_per_row() -> None:
    """Token z písmen přežije tokenizaci i normalizaci čísel a v každé zprávě je jiný (DF = 1)."""
    rows = synthetic_rows(
        "discharge", NOTE_SHAPE, "eda-sentinel", 30, text_chars=TEXT_CHARS, codes=CODES
    )

    tokens = [row["text"].split()[-1] for row in rows]

    assert all(re.fullmatch(r"edasentinel[a-z]+", token) for token in tokens)
    assert len(set(tokens)) == len(tokens)


def test_default_rows_are_four_blocks_and_rare_rows() -> None:
    """Výchozí počet řádků jsou čtyři bloky po MIN_CELL a vzácné řádky."""
    assert DEFAULT_ROWS == 4 * MIN_CELL + RARE_ROWS


@pytest.mark.parametrize(("key", "values"), EXPECTED.items())
def test_typed_column_has_block_values_and_rare_rows_follow_rule(
    key: tuple[str, str], values: tuple[str, ...]
) -> None:
    """Typový sloupec nese hodnoty bloků (neprázdná množina), vzácné řádky jen ve svědcích prefix."""
    table, column = key
    cells = _column(table, column)

    for block in range(4):
        assert (
            cells[block * MIN_CELL : (block + 1) * MIN_CELL]
            == [values[block]] * MIN_CELL
        )
    rare = cells[4 * MIN_CELL :]
    assert len(rare) == RARE_ROWS
    if key in WITNESSES:
        assert all(cell.startswith(PREFIX) for cell in rare)
    else:
        assert rare == [values[0]] * RARE_ROWS


@pytest.mark.parametrize("table", TYPED_TABLES)
def test_values_without_prefix_are_shared_by_min_cell_rows(table: str) -> None:
    """Pravidlo kanárka: kategorie bez prefixu má aspoň MIN_CELL výskytů, ostatní nesou prefix."""
    numeric = {"anchor_age", "anchor_year", "admittime"}
    for column in COLUMNS[table]:
        if column in numeric:
            continue
        cells = _column(table, column)
        bare = [cell for cell in cells if PREFIX not in cell]
        assert all(bare.count(cell) >= MIN_CELL for cell in set(bare)), column
        if (table, column) not in EXPECTED:
            assert not bare, column


def test_patient_age_at_admission_matches_blocks() -> None:
    """Věk při přijetí je 20 + řádek % 40 a desetiletá pásma sedí na bloky."""
    patients = _table("patients")
    admissions = _table("admissions")
    ages = [
        int(patient["anchor_age"])
        + int(admission["admittime"][:4])
        - int(patient["anchor_year"])
        for patient, admission in zip(patients, admissions, strict=True)
    ]

    assert ages == [20 + row % 40 for row in range(DEFAULT_ROWS)]
    bands = [age // 10 * 10 for age in ages]
    assert [bands.count(band) for band in (20, 30, 40, 50)] == [12, 10, 10, 10]


def test_era_follows_version_of_same_row() -> None:
    """Období pacienta sleduje verzi ICD diagnózy téhož řádku."""
    eras = _column("patients", "anchor_year_group")
    versions = _column("diagnoses_icd", "icd_version")

    pairs = {
        (era, version)
        for era, version in zip(eras, versions, strict=True)
        if PREFIX not in era
    }
    assert pairs == {("2008 - 2010", "9"), ("2017 - 2019", "10")}


def test_dictionary_holds_block_codes_with_versions_and_titles() -> None:
    """Číselník: řádky 0-3 nesou kódy bloků s verzí a názvem bez prefixu, ostatní prefix."""
    rows = _table("d_icd_diagnoses")

    assert [row["icd_code"] for row in rows[:4]] == list(BLOCK_CODES)
    assert [row["icd_version"] for row in rows[:4]] == ["9", "9", "10", "10"]
    assert [row["long_title"] for row in rows[:4]] == [
        "diabetes ICD-9",
        "copd ICD-9",
        "diabetes ICD-10",
        "copd ICD-10",
    ]
    assert all(PREFIX in cell for row in rows[4:] for cell in row.values())


def test_block_codes_are_first_prefixes_of_the_config() -> None:
    """Kód bloku je první předpona z configu, ne natvrdo zapsaný kód."""
    other = {
        diagnosis: {
            version: tuple(f"Z{index}" for index in range(2))
            if diagnosis is Diagnosis.DIABETES and version is IcdVersion.TEN
            else prefixes
            for version, prefixes in by_version.items()
        }
        for diagnosis, by_version in PREFIXES.items()
    }
    rows = synthetic_rows(
        "diagnoses_icd",
        TableShape(records=1, columns=("icd_code",)),
        PREFIX,
        3 * MIN_CELL,
        text_chars=TEXT_CHARS,
        codes=DiagnosisCodes(prefixes=other),
    )

    assert rows[2 * MIN_CELL]["icd_code"] == "Z0"


def test_fewer_rows_give_partial_blocks() -> None:
    """Menší počet řádků dá bloky jen částečně a nespadne."""
    assert _column("diagnoses_icd", "icd_version", 12) == ["9"] * 10 + ["9"] * 2


def test_table_without_typed_columns_is_unchanged() -> None:
    """Tabulka bez typových sloupců se nemění, buňky jsou jedinečné s prefixem."""
    rows = synthetic_rows(
        "discharge", SHAPE, PREFIX, DEFAULT_ROWS, text_chars=TEXT_CHARS, codes=CODES
    )
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all(PREFIX in cell for cell in cells)


@pytest.mark.parametrize(
    ("diagnosis", "version"),
    [(Diagnosis.DIABETES, IcdVersion.NINE), (Diagnosis.COPD, IcdVersion.TEN)],
)
def test_block_diagnosis_without_prefix_is_refused(
    diagnosis: Diagnosis, version: IcdVersion
) -> None:
    """Diagnóza bloku bez předpony v některé verzi je InvalidInputError."""
    prefixes = {
        d: {v: p for v, p in by_version.items() if (d, v) != (diagnosis, version)}
        for d, by_version in PREFIXES.items()
    }

    with pytest.raises(InvalidInputError, match=diagnosis.value):
        synthetic_rows(
            "patients",
            TableShape(records=1, columns=("subject_id",)),
            PREFIX,
            1,
            text_chars=TEXT_CHARS,
            codes=DiagnosisCodes(prefixes=prefixes),
        )
