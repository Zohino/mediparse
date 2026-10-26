"""Syntetické řádky tabulky ve tvaru MIMIC: buňky s prefixem, typové hodnoty po blocích a pravidlo kanárka."""

from __future__ import annotations

import re
from typing import Final

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.dua import MIN_CELL
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.section_headers import SectionLexicon
from mediparse.domain.synthetic_discharge import (
    BHC_SECTION,
    synthetic_discharge,
    text_token,
)
from mediparse.domain.synthetic_tables import (
    BLOCKS,
    DEFAULT_ROWS,
    RARE_ROWS,
    SyntheticRun,
    synthetic_rows,
)
from tests.cohort_support import CODES, PREFIXES
from tests.section_support import REPOSITORY_LEXICON

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
    ("patients", "anchor_age"): ("15", "25", "35", "45"),
    ("patients", "anchor_year"): ("2150",) * 4,
    ("admissions", "admittime"): ("2155-01-01 00:00:00",) * 4,
}
WITNESSES: Final = {
    ("patients", "anchor_year_group"),
    ("diagnoses_icd", "icd_version"),
    ("diagnoses_icd", "icd_code"),
}
TYPED_TABLES: Final = ("patients", "admissions", "diagnoses_icd")


def _run(prefix: str, rows: int, codes: DiagnosisCodes = CODES) -> SyntheticRun:
    return SyntheticRun(
        prefix=prefix,
        rows=rows,
        text_chars=TEXT_CHARS,
        codes=codes,
        lexicon=REPOSITORY_LEXICON,
    )


def _table(table: str, rows: int = DEFAULT_ROWS) -> list[dict[str, str]]:
    shape = TableShape(records=rows, columns=COLUMNS[table])
    return synthetic_rows(table, shape, _run(PREFIX, rows))


def _column(table: str, column: str, rows: int = DEFAULT_ROWS) -> list[str]:
    return [row[column] for row in _table(table, rows)]


def test_rows_have_requested_count_and_columns_in_manifest_order() -> None:
    """Každý řádek má všechny sloupce v pořadí manifestu a řádků je tolik, kolik se žádá."""
    rows = synthetic_rows("discharge", SHAPE, _run("sentinel", ROWS))

    assert len(rows) == ROWS
    assert all(tuple(row) == SHAPE.columns for row in rows)


def test_cells_are_unique_within_table_and_carry_prefix() -> None:
    """Každá buňka tabulky je v ní jedinečná a obsahuje prefix, takže únik zachytí hledání prefixu."""
    rows = synthetic_rows("discharge", SHAPE, _run("sentinel", ROWS))
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all("sentinel" in cell for cell in cells)


def test_cells_name_table_column_and_row() -> None:
    """Buňka je ve tvaru prefix-tabulka-sloupec-řádek."""
    rows = synthetic_rows("radiology", NOTE_SHAPE, _run("sentinel", 2))

    assert rows[1]["note_id"] == "sentinel-radiology-note_id-1"


def test_rows_are_deterministic() -> None:
    """Stejné vstupy dávají stejné řádky."""
    assert synthetic_rows("t", SHAPE, _run("p", 3)) == synthetic_rows(
        "t", SHAPE, _run("p", 3)
    )


def test_zero_rows_are_refused() -> None:
    """Nulový počet řádků je chyba."""
    with pytest.raises(InvalidInputError, match="řádků"):
        synthetic_rows("t", SHAPE, _run("p", 0))


def test_table_without_columns_is_refused() -> None:
    """Tabulka bez sloupců je chyba a jmenuje tabulku."""
    with pytest.raises(InvalidInputError, match="empty"):
        synthetic_rows("empty", TableShape(records=0, columns=()), _run("p", 3))


def test_empty_prefix_is_refused() -> None:
    """Prázdný prefix je chyba, protože kanárek by nic nehledal."""
    with pytest.raises(InvalidInputError, match="Prefix"):
        synthetic_rows("t", SHAPE, _run("", ROWS))


def test_patient_and_admission_keys_are_shared_across_tables() -> None:
    """Řádek r každé tabulky patří pacientovi a hospitalizaci r, takže se tabulky spojí jako v MIMIC."""
    notes = synthetic_rows("discharge", NOTE_SHAPE, _run("p", ROWS))
    codes = synthetic_rows(
        "diagnoses_icd",
        TableShape(records=5, columns=("subject_id", "hadm_id", "icd_code")),
        _run("p", ROWS),
    )

    assert [row["hadm_id"] for row in notes] == [row["hadm_id"] for row in codes]
    assert [row["subject_id"] for row in notes] == [row["subject_id"] for row in codes]
    assert notes[3]["hadm_id"] == "p-admissions-hadm_id-3"
    assert notes[3]["subject_id"] == "p-patients-subject_id-3"


def test_bhc_note_id_refers_to_discharge_and_radiology_keeps_its_own() -> None:
    """BHC se páruje s propouštěcí zprávou přes note_id, radiologie má note_id vlastní jako v MIMIC."""
    shape = TableShape(records=5, columns=("note_id", "input"))

    bhc = synthetic_rows("mimic-iv-bhc", shape, _run("p", 2))
    radiology = synthetic_rows("radiology", NOTE_SHAPE, _run("p", 2))

    assert bhc[1]["note_id"] == "p-discharge-note_id-1"
    assert radiology[1]["note_id"] == "p-radiology-note_id-1"


def test_text_starts_with_its_cell_and_reaches_minimum_length() -> None:
    """Text začíná svou buňkou (u zpráv bloků samostatným řádkem) a má aspoň požadovaný počet znaků."""
    rows = synthetic_rows("discharge", NOTE_SHAPE, _run("p", DEFAULT_ROWS))

    for index, row in enumerate(rows):
        cell = f"p-discharge-text-{index}"
        separator = "\n" if index < BLOCKS * MIN_CELL else " "
        assert row["text"].startswith(cell + separator)
        assert len(row["text"]) >= TEXT_CHARS


def test_text_token_is_letters_only_and_unique_per_row() -> None:
    """Token z písmen přežije tokenizaci a normalizaci čísel a v každé zprávě je jiný (DF = 1)."""
    rows = synthetic_rows("discharge", NOTE_SHAPE, _run("eda-sentinel", 30))

    tokens = [set(re.findall(r"edasentinel[a-z]+", row["text"])) for row in rows]

    assert all(len(found) == 1 for found in tokens)
    assert len({next(iter(found)) for found in tokens}) == len(tokens)


def test_block_texts_and_targets_come_from_the_discharge_module() -> None:
    """Text propouštěcí zprávy a target BHC bloků jdou ze ``synthetic_discharge``, jedna zpráva na řádek."""
    run = _run(PREFIX, DEFAULT_ROWS)
    texts = synthetic_rows("discharge", NOTE_SHAPE, run)
    targets = synthetic_rows(
        "mimic-iv-bhc", TableShape(records=5, columns=("note_id", "target")), run
    )

    for row in range(BLOCKS * MIN_CELL):
        note = synthetic_discharge(
            PREFIX, REPOSITORY_LEXICON, row, row // MIN_CELL, TEXT_CHARS
        )
        assert texts[row]["text"] == note.text
        assert targets[row]["target"] == note.bhc_target
        assert text_token(PREFIX, row) in note.text


def test_rare_rows_keep_text_and_target_as_cells_without_headings() -> None:
    """Vzácné řádky zůstávají bez nadpisů, jejich target je buňka."""
    run = _run(PREFIX, DEFAULT_ROWS)
    texts = synthetic_rows("discharge", NOTE_SHAPE, run)
    targets = synthetic_rows(
        "mimic-iv-bhc", TableShape(records=5, columns=("note_id", "target")), run
    )

    for row in range(BLOCKS * MIN_CELL, DEFAULT_ROWS):
        assert texts[row]["text"].count("\n") == 0
        assert targets[row]["target"] == f"{PREFIX}-mimic-iv-bhc-target-{row}"


def test_radiology_text_is_unchanged() -> None:
    """Radiologie a ostatní textové sloupce zůstávají buňkou a tokeny."""
    rows = synthetic_rows("radiology", NOTE_SHAPE, _run("p", 3))

    assert rows[1]["text"].startswith("p-radiology-text-1 ")
    assert "\n" not in rows[1]["text"]


def test_lexicon_without_bhc_section_is_refused() -> None:
    """Lexikon bez sekce BHC je InvalidInputError s klíčem sekce."""
    sections = {
        key: section
        for key, section in REPOSITORY_LEXICON.sections.items()
        if key != BHC_SECTION
    }

    with pytest.raises(InvalidInputError, match=BHC_SECTION):
        SyntheticRun(
            prefix=PREFIX,
            rows=1,
            text_chars=TEXT_CHARS,
            codes=CODES,
            lexicon=SectionLexicon(
                sections=sections, preamble=REPOSITORY_LEXICON.preamble
            ),
        )


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
    for column in COLUMNS[table]:
        cells = _column(table, column)
        bare = [cell for cell in cells if PREFIX not in cell]
        assert all(bare.count(cell) >= MIN_CELL for cell in set(bare)), column
        if (table, column) not in EXPECTED:
            assert not bare, column


def test_patient_age_at_admission_is_shared_within_block() -> None:
    """Věk při přijetí je 20/30/40/50 podle bloku a vzácné řádky mají věk bloku 0."""
    patients = _table("patients")
    admissions = _table("admissions")
    ages = [
        int(patient["anchor_age"])
        + int(admission["admittime"][:4])
        - int(patient["anchor_year"])
        for patient, admission in zip(patients, admissions, strict=True)
    ]

    expected = [20 + 10 * ((row // MIN_CELL) % BLOCKS) for row in range(DEFAULT_ROWS)]
    assert ages == expected
    assert [ages.count(age) for age in (20, 30, 40, 50)] == [
        MIN_CELL + RARE_ROWS,
        MIN_CELL,
        MIN_CELL,
        MIN_CELL,
    ]


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
        _run(PREFIX, 3 * MIN_CELL, DiagnosisCodes(prefixes=other)),
    )

    assert rows[2 * MIN_CELL]["icd_code"] == "Z0"


def test_fewer_rows_give_partial_blocks() -> None:
    """Menší počet řádků dá bloky jen částečně a nespadne."""
    assert _column("patients", "gender", 12) == ["F"] * MIN_CELL + ["M"] * 2


def test_table_without_typed_columns_is_unchanged() -> None:
    """Tabulka bez typových sloupců se nemění, buňky jsou jedinečné s prefixem."""
    rows = synthetic_rows("discharge", SHAPE, _run(PREFIX, DEFAULT_ROWS))
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

    with pytest.raises(
        InvalidInputError, match=rf"{diagnosis.value}.*ICD-{version.value}"
    ):
        _run(PREFIX, 1, DiagnosisCodes(prefixes=prefixes))
