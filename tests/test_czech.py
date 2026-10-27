"""Český zápis celých čísel pro text práce."""

import pytest

from mediparse.domain import czech
from mediparse.domain.czech import NBSP, format_number
from mediparse.domain.dua import CategoryCount


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1\u00a0000"),
        (2726465013, "2\u00a0726\u00a0465\u00a0013"),
        (-1569, "-1\u00a0569"),
    ],
)
def test_format_number(value: int, expected: str) -> None:
    """Oddělovač tisíců je nezlomitelná mezera i nad rozsahem R integeru."""
    assert format_number(value) == expected


def test_nbsp_is_nonbreaking() -> None:
    """Sdílená konstanta pro český zápis je nezlomitelná mezera U+00A0."""
    assert NBSP == "\u00a0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12.46, "12,5"),
        (100, "100,0"),
        (1, "1,0"),
        (2.25, "2,2"),
        (0.05, "0,0"),
        (0.15, "0,1"),
        (429.05, "429,0"),
        (7.45, "7,4"),
        (79.55, "79,6"),
        (-5550.95, "-5551,0"),
        (-389.95, "-390,0"),
        (-0.04, "0,0"),
        (99.95, "100,0"),
        (1234.56, "1234,6"),
        (10000000.25, "10000000,2"),
        (0, "0,0"),
        (-0.0, "0,0"),
        (1e15, "1000000000000000,0"),
    ],
)
def test_decimal_text(value: float, expected: str) -> None:
    """Desetinné hodnoty odpovídají round z R, včetně hraničních případů."""
    assert czech.decimal_text(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1234567, "1\u00a0234\u00a0567"), (5, "5"), (None, "\u2013")],
)
def test_number_text(value: int | None, expected: str) -> None:
    """Chybějící počet nahradí pomlčka, tisíce oddělí NBSP."""
    assert czech.number_text(value) == expected


def test_missing_text() -> None:
    """Chybějící hodnoty mají jednotnou typografickou pomlčku."""
    assert czech.MISSING == "\u2013"


@pytest.mark.parametrize(
    ("n", "total", "expected"),
    [
        (3, 1000, "0,3"),
        (25, 200, "12,5"),
        (15, 0, "\u2013"),
        (15, -1, "\u2013"),
        (0, 100, "0,0"),
        (20, 100, "20,0"),
        (22, 331604, "<\u00a00,1"),
        (0, 331604, "0,0"),
        (2, 1000, "0,2"),
        (1, 2000, "<\u00a00,1"),
        (1, 3, "33,3"),
        (5, 10000, "<\u00a00,1"),
        (499, 1000, "49,9"),
        (2, 7, "28,6"),
    ],
)
def test_percent_text(n: int, total: int, expected: str) -> None:
    """Malé počty se zveřejní a nenulový podíl se nezobrazí jako nula."""
    assert czech.percent_text(n, total) == expected


@pytest.mark.parametrize(
    ("n", "total", "expected"),
    [
        (22, 331604, "<\u00a00,1\u00a0%"),
        (3, 1000, "0,3\u00a0%"),
        (25, 200, "12,5\u00a0%"),
        (15, 0, "\u2013"),
    ],
)
def test_percent_label(n: int, total: int, expected: str) -> None:
    """Procento oddělí nezlomitelná mezera; pomlčka zůstane samostatná."""
    assert czech.percent_label(n, total) == expected


def test_share_rows() -> None:
    """Podíly z celku zachovají všechny kategorie včetně malých počtů."""
    rows = (
        CategoryCount(("1",), 1990),
        CategoryCount(("2",), 6),
        CategoryCount(("3+",), 4),
    )
    assert czech.share_rows(rows) == (
        ("1", "1\u00a0990", "99,5"),
        ("2", "6", "0,3"),
        ("3+", "4", "0,2"),
    )


def test_share_rows_with_multiple_labels_and_zero_total() -> None:
    """Textová tabulka zachová všechny popisky a zvládne nulový celek."""
    assert czech.share_rows((CategoryCount(("A", "x"), 0),)) == (
        ("A", "x", "0", "\u2013"),
    )


def test_empty_share_rows() -> None:
    """Prázdná tabulka má prázdný textový výstup."""
    assert czech.share_rows(()) == ()
