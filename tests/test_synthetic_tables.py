"""Syntetické řádky tabulky ve tvaru MIMIC: jedinečné buňky, v nichž je vidět prefix."""

from __future__ import annotations

from typing import Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.synthetic_tables import synthetic_rows

ROWS: Final = 12
SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "text"))


def test_rows_have_requested_count_and_columns_in_manifest_order() -> None:
    """Každý řádek má všechny sloupce v pořadí manifestu a řádků je tolik, kolik se žádá."""
    rows = synthetic_rows("discharge", SHAPE, "sentinel", ROWS)

    assert len(rows) == ROWS
    assert all(tuple(row) == SHAPE.columns for row in rows)


def test_cells_are_unique_across_table_and_carry_prefix() -> None:
    """Každá buňka tabulky je jedinečná a obsahuje prefix, takže únik zachytí hledání prefixu."""
    rows = synthetic_rows("discharge", SHAPE, "sentinel", ROWS)
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all("sentinel" in cell for cell in cells)


def test_cells_name_table_column_and_row() -> None:
    """Buňka je ve tvaru prefix-tabulka-sloupec-řádek."""
    rows = synthetic_rows("discharge", SHAPE, "sentinel", 2)

    assert rows[1]["text"] == "sentinel-discharge-text-1"


def test_rows_are_deterministic() -> None:
    """Stejné vstupy dávají stejné řádky."""
    assert synthetic_rows("t", SHAPE, "p", 3) == synthetic_rows("t", SHAPE, "p", 3)


def test_zero_rows_are_refused() -> None:
    """Nulový počet řádků je chyba."""
    with pytest.raises(InvalidInputError, match="řádků"):
        synthetic_rows("t", SHAPE, "p", 0)


def test_table_without_columns_is_refused() -> None:
    """Tabulka bez sloupců je chyba a jmenuje tabulku."""
    with pytest.raises(InvalidInputError, match="empty"):
        synthetic_rows("empty", TableShape(records=0, columns=()), "p", 3)


def test_empty_prefix_is_refused() -> None:
    """Prázdný prefix je chyba, protože kanárek by nic nehledal."""
    with pytest.raises(InvalidInputError, match="Prefix"):
        synthetic_rows("t", SHAPE, "", ROWS)
