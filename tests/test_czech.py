"""Český zápis celých čísel pro text práce."""

import pytest

from mediparse.domain.czech import NBSP, format_number


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
    assert NBSP == "\u00a0"
    assert format_number(value) == expected
