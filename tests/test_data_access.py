"""Pravidlo přístupu k datům MIMIC: kdo je smí číst a s jakým důvodem odmítnutí."""

from __future__ import annotations

import pytest

from mediparse.domain.data_access import (
    BLOCKING_VARIABLES,
    DataAccessError,
    blocked_reason,
    check_data_access,
)
from mediparse.domain.inputs import InvalidInputError

LOCATION = "/data/parquet"
MARKER = "znacka.json"


def test_blocking_variables_are_ci_then_claude_code() -> None:
    """Pořadí proměnných určuje, který důvod se nahlásí první."""
    assert list(BLOCKING_VARIABLES) == ["CI", "CLAUDECODE"]


@pytest.mark.parametrize("name", ["CI", "CLAUDECODE"])
def test_blocked_reason_names_the_variable_reason(name: str) -> None:
    """Každá blokující proměnná vrátí svůj důvod."""
    assert blocked_reason({name: "1"}) == BLOCKING_VARIABLES[name]


def test_blocked_reason_prefers_ci() -> None:
    """Při obou proměnných vyhrává CI."""
    assert blocked_reason({"CI": "true", "CLAUDECODE": "1"}) == BLOCKING_VARIABLES["CI"]


@pytest.mark.parametrize("environ", [{}, {"CI": "", "CLAUDECODE": ""}, {"JINE": "1"}])
def test_blocked_reason_is_none_without_non_empty_variable(
    environ: dict[str, str],
) -> None:
    """Prázdná nebo chybějící proměnná nic neblokuje."""
    assert blocked_reason(environ) is None


@pytest.mark.parametrize(
    ("environ", "words"),
    [({"CI": "true"}, "CI"), ({"CLAUDECODE": "1"}, "Claude Code")],
    ids=["ci", "claude-code"],
)
def test_check_refuses_real_data_in_blocked_environment(
    environ: dict[str, str], words: str
) -> None:
    """Skutečná data v blokujícím prostředí hláška jmenuje místo, značku i důvod."""
    with pytest.raises(DataAccessError) as refusal:
        check_data_access(environ, synthetic=False, location=LOCATION, marker=MARKER)

    message = str(refusal.value)
    assert LOCATION in message
    assert MARKER in message
    assert words in message


def test_refusal_is_an_invalid_input() -> None:
    """Odmítnutí přístupu je odmítnutí kroku."""
    assert issubclass(DataAccessError, InvalidInputError)


def test_check_allows_synthetic_data_in_blocked_environment() -> None:
    """Syntetická data projdou i v obou blokujících prostředích."""
    check_data_access(
        {"CI": "true", "CLAUDECODE": "1"},
        synthetic=True,
        location=LOCATION,
        marker=MARKER,
    )


def test_check_allows_real_data_without_blocking_variable() -> None:
    """Bez blokující proměnné projdou i data bez značky."""
    check_data_access({}, synthetic=False, location=LOCATION, marker=MARKER)
