"""Pravidlo přístupu k datům MIMIC: kdo je smí číst a s jakým důvodem odmítnutí."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mediparse.domain.data_access import (
    BlockingVariable,
    DataAccessError,
    DataAccessRules,
    blocked_reason,
    check_data_access,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.data_access_rules import load_data_access_rules

if TYPE_CHECKING:
    from pathlib import Path

LOCATION = "/data/parquet"
MARKER = "znacka.json"


def test_blocking_variables_are_ci_then_claude_code() -> None:
    """Pořadí proměnných určuje, který důvod se nahlásí první."""
    assert [v.name for v in load_data_access_rules().blocking] == ["CI", "CLAUDECODE"]


@pytest.mark.parametrize("name", ["CI", "CLAUDECODE"])
def test_blocked_reason_names_the_variable_reason(name: str) -> None:
    """Každá blokující proměnná vrátí svůj důvod."""
    assert blocked_reason({name: "1"}, load_data_access_rules()) == next(
        v.reason for v in load_data_access_rules().blocking if v.name == name
    )


def test_blocked_reason_prefers_ci() -> None:
    """Při obou proměnných vyhrává CI."""
    assert (
        blocked_reason({"CI": "true", "CLAUDECODE": "1"}, load_data_access_rules())
        == load_data_access_rules().blocking[0].reason
    )


@pytest.mark.parametrize("environ", [{}, {"CI": "", "CLAUDECODE": ""}, {"JINE": "1"}])
def test_blocked_reason_is_none_without_non_empty_variable(
    environ: dict[str, str],
) -> None:
    """Prázdná nebo chybějící proměnná nic neblokuje."""
    assert blocked_reason(environ, load_data_access_rules()) is None


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
        check_data_access(
            environ,
            load_data_access_rules(),
            synthetic=False,
            location=LOCATION,
            marker=MARKER,
        )

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
        load_data_access_rules(),
        synthetic=True,
        location=LOCATION,
        marker=MARKER,
    )


def test_check_allows_real_data_without_blocking_variable() -> None:
    """Bez blokující proměnné projdou i data bez značky."""
    check_data_access(
        {}, load_data_access_rules(), synthetic=False, location=LOCATION, marker=MARKER
    )


def test_rules_come_from_data() -> None:
    """I nová proměnná bez změny funkce blokuje podle vložených pravidel."""
    rules = DataAccessRules(blocking=(BlockingVariable(name="CODEX", reason="x"),))
    assert blocked_reason({"CODEX": "1"}, rules) == "x"


@pytest.mark.parametrize(
    "content",
    [
        '{"blocking": []}',
        '{"blocking": [{"name":"CI","reason":"x"},{"name":"CI","reason":"y"}]}',
        '{"blocking": [{"name":"CI","reason":""}]}',
        '{"blocking": [{"name":"","reason":"x"}]}',
        '{"blocking": [{"name":"CI","reason":"x","extra":1}]}',
        '{"blocking": [{"name":"CI","reason":"x"}],"extra":1}',
        '{"blocking": [{"name":1,"reason":"x"}]}',
    ],
)
def test_invalid_rules_are_refused(content: str) -> None:
    """Prázdná, duplicitní i nestriktní pravidla nejsou bezpečnostní pravidlo."""
    with pytest.raises(ValidationError):
        DataAccessRules.model_validate_json(content)


@pytest.mark.parametrize("content", [None, "", "{}", "ne-JSON"])
def test_loader_refuses_missing_or_invalid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    """Poškozená či chybějící pravidla se přeloží na doménové odmítnutí."""
    if content is not None:
        (tmp_path / "data_access.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "mediparse.infrastructure.data_access_rules.files", lambda _: tmp_path
    )
    with pytest.raises(InvalidInputError, match=r"data_access\.json"):
        load_data_access_rules()
