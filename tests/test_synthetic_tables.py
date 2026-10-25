"""Syntetické řádky tabulky ve tvaru MIMIC: jedinečné buňky, v nichž je vidět prefix."""

from __future__ import annotations

import re
from typing import Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.synthetic_tables import synthetic_rows

ROWS: Final = 12
TEXT_CHARS: Final = 100
SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "text"))
NOTE_SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "hadm_id", "text"))


def test_rows_have_requested_count_and_columns_in_manifest_order() -> None:
    """Každý řádek má všechny sloupce v pořadí manifestu a řádků je tolik, kolik se žádá."""
    rows = synthetic_rows("discharge", SHAPE, "sentinel", ROWS, text_chars=TEXT_CHARS)

    assert len(rows) == ROWS
    assert all(tuple(row) == SHAPE.columns for row in rows)


def test_cells_are_unique_within_table_and_carry_prefix() -> None:
    """Každá buňka tabulky je v ní jedinečná a obsahuje prefix, takže únik zachytí hledání prefixu."""
    rows = synthetic_rows("discharge", SHAPE, "sentinel", ROWS, text_chars=TEXT_CHARS)
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all("sentinel" in cell for cell in cells)


def test_cells_name_table_column_and_row() -> None:
    """Buňka je ve tvaru prefix-tabulka-sloupec-řádek."""
    rows = synthetic_rows("radiology", NOTE_SHAPE, "sentinel", 2, text_chars=TEXT_CHARS)

    assert rows[1]["note_id"] == "sentinel-radiology-note_id-1"


def test_rows_are_deterministic() -> None:
    """Stejné vstupy dávají stejné řádky."""
    assert synthetic_rows("t", SHAPE, "p", 3, text_chars=TEXT_CHARS) == synthetic_rows(
        "t", SHAPE, "p", 3, text_chars=TEXT_CHARS
    )


def test_zero_rows_are_refused() -> None:
    """Nulový počet řádků je chyba."""
    with pytest.raises(InvalidInputError, match="řádků"):
        synthetic_rows("t", SHAPE, "p", 0, text_chars=TEXT_CHARS)


def test_table_without_columns_is_refused() -> None:
    """Tabulka bez sloupců je chyba a jmenuje tabulku."""
    with pytest.raises(InvalidInputError, match="empty"):
        synthetic_rows(
            "empty", TableShape(records=0, columns=()), "p", 3, text_chars=TEXT_CHARS
        )


def test_empty_prefix_is_refused() -> None:
    """Prázdný prefix je chyba, protože kanárek by nic nehledal."""
    with pytest.raises(InvalidInputError, match="Prefix"):
        synthetic_rows("t", SHAPE, "", ROWS, text_chars=TEXT_CHARS)


def test_patient_and_admission_keys_are_shared_across_tables() -> None:
    """Řádek r každé tabulky patří pacientovi a hospitalizaci r, takže se tabulky spojí jako v MIMIC."""
    notes = synthetic_rows("discharge", NOTE_SHAPE, "p", ROWS, text_chars=TEXT_CHARS)
    codes = synthetic_rows(
        "diagnoses_icd",
        TableShape(records=5, columns=("subject_id", "hadm_id", "icd_code")),
        "p",
        ROWS,
        text_chars=TEXT_CHARS,
    )

    assert [row["hadm_id"] for row in notes] == [row["hadm_id"] for row in codes]
    assert [row["subject_id"] for row in notes] == [row["subject_id"] for row in codes]
    assert notes[3]["hadm_id"] == "p-admissions-hadm_id-3"
    assert notes[3]["subject_id"] == "p-patients-subject_id-3"


def test_bhc_note_id_refers_to_discharge_and_radiology_keeps_its_own() -> None:
    """BHC se páruje s propouštěcí zprávou přes note_id, radiologie má note_id vlastní jako v MIMIC."""
    shape = TableShape(records=5, columns=("note_id", "input"))

    bhc = synthetic_rows("mimic-iv-bhc", shape, "p", 2, text_chars=TEXT_CHARS)
    radiology = synthetic_rows("radiology", NOTE_SHAPE, "p", 2, text_chars=TEXT_CHARS)

    assert bhc[1]["note_id"] == "p-discharge-note_id-1"
    assert radiology[1]["note_id"] == "p-radiology-note_id-1"


def test_text_starts_with_its_cell_and_reaches_minimum_length() -> None:
    """Text začíná svou buňkou a má aspoň požadovaný počet znaků, aby prošel filtrem délky."""
    rows = synthetic_rows("discharge", NOTE_SHAPE, "p", ROWS, text_chars=TEXT_CHARS)

    assert all(
        row["text"].startswith(f"p-discharge-text-{index} ")
        for index, row in enumerate(rows)
    )
    assert all(len(row["text"]) >= TEXT_CHARS for row in rows)


def test_text_token_is_letters_only_and_unique_per_row() -> None:
    """Token z písmen přežije tokenizaci i normalizaci čísel a v každé zprávě je jiný (DF = 1)."""
    rows = synthetic_rows(
        "discharge", NOTE_SHAPE, "eda-sentinel", 30, text_chars=TEXT_CHARS
    )

    tokens = [row["text"].split()[-1] for row in rows]

    assert all(re.fullmatch(r"edasentinel[a-z]+", token) for token in tokens)
    assert len(set(tokens)) == len(tokens)
