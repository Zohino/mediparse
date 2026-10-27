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
        (0.25, "0,2"),
        (0.05, "0,1"),
        (-0.04, "0,0"),
        (1234.56, "1\u00a0234,6"),
        (10000000.25, "10\u00a0000\u00a0000,2"),
        (-1234.56, "-1\u00a0234,6"),
        (0, "0,0"),
        (-0.0, "0,0"),
    ],
)
def test_decimal_text(value: float, expected: str) -> None:
    """Python zaokrouhlí k sudé; binární 0.05 leží nad polovinou a dá 0,1."""
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
        (1, 2000, "0,1"),
        (1, 3, "33,3"),
        (5, 10000, "0,1"),
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


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "zmínka"),
        (2, "zmínky"),
        (3, "zmínky"),
        (4, "zmínky"),
        (0, "zmínek"),
        (5, "zmínek"),
        (11, "zmínek"),
        (12, "zmínek"),
        (21, "zmínek"),
        (22, "zmínek"),
        (100, "zmínek"),
        (1001, "zmínek"),
        (-1, "zmínek"),
        (-2, "zmínek"),
        (1.5, "zmínky"),
        (-1.5, "zmínky"),
    ],
)
def test_plural_form(n: float, expected: str) -> None:
    """Celé i necelé počty dostanou příslušný český tvar."""
    forms = czech.PluralForms("zmínka", "zmínky", "zmínek")
    assert czech.plural_form(n, forms) == expected


@pytest.mark.parametrize(
    ("n", "expected"), [(1, "den"), (2, "dny"), (5, "dnů"), (1.5, "dne")]
)
def test_plural_form_with_fraction(n: float, expected: str) -> None:
    """Volitelný čtvrtý tvar platí jen pro desetinná čísla."""
    assert (
        czech.plural_form(n, czech.PluralForms("den", "dny", "dnů", "dne")) == expected
    )


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (12345.678, "12\u00a0345,68"),
        (1e-5, "0,00001"),
        (2.5e-7, "0,00000025"),
        (1234567.5, "1\u00a0234\u00a0568"),
        (123456.75, "123\u00a0456,8"),
        (1234.5, "1\u00a0234,5"),
        (-1.5, "-1,5"),
        (0, "0"),
        (123456789012, "123\u00a0456\u00a0789\u00a0012"),
        (-1234, "-1\u00a0234"),
        (12345678.5, "12\u00a0345\u00a0678"),
        (98765432.123, "98\u00a0765\u00a0432"),
        (9999999.5, "10\u00a0000\u00a0000"),
    ],
)
def test_number_phrase(n: float, expected: str) -> None:
    """Sedm platných číslic odpovídá sondě R bez vědeckého zápisu."""
    assert czech.number_phrase(n) == expected


@pytest.mark.parametrize(
    ("n", "zero", "expected"),
    [
        (5, None, "5\u00a0zmínek"),
        (1, None, "1\u00a0zmínka"),
        (12345, None, "12\u00a0345\u00a0zmínek"),
        (1.5, None, "1,5\u00a0zmínky"),
        (0, "žádná zmínka", "žádná zmínka"),
        (3, "žádná zmínka", "3\u00a0zmínky"),
        (0, None, "0\u00a0zmínek"),
        (0, "", ""),
    ],
)
def test_count_phrase(n: float, zero: str | None, expected: str) -> None:
    """Počet a skloňovaný tvar spojí NBSP, nulu může nahradit text."""
    forms = czech.PluralForms("zmínka", "zmínky", "zmínek")
    assert czech.count_phrase(n, forms, zero=zero) == expected


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ((), "žádný"),
        (("a",), "a"),
        (("a", "b"), "a a b"),
        (("a", "b", "c"), "a, b a c"),
    ],
)
def test_join_labels(labels: tuple[str, ...], expected: str) -> None:
    """Výčet končí spojkou a; prázdný dostane pojmenování."""
    assert czech.join_labels(labels) == expected
