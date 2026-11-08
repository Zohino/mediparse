"""Pravidlo přístupu k datům MIMIC: kdo je smí číst a s jakým důvodem odmítnutí."""

from __future__ import annotations

import json
from operator import itemgetter
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
from tests.support import REPOSITORY, blocking_reason

if TYPE_CHECKING:
    from pathlib import Path

LOCATION = "/data/parquet"
MARKER = "znacka.json"


def test_blocking_variables_are_ci_then_claude_code_then_codex() -> None:
    """Pořadí proměnných určuje, který důvod se nahlásí první."""
    assert [v.name for v in load_data_access_rules().blocking] == [
        "CI",
        "CLAUDECODE",
        "CODEX_THREAD_ID",
    ]


@pytest.mark.parametrize(
    "name", [variable.name for variable in load_data_access_rules().blocking]
)
def test_blocked_reason_names_the_variable_reason(name: str) -> None:
    """Každá blokující proměnná vrátí svůj důvod."""
    assert blocked_reason({name: "1"}, load_data_access_rules()) == blocking_reason(
        name
    )


def test_blocked_reason_prefers_ci() -> None:
    """Při všech proměnných vyhrává CI."""
    assert (
        blocked_reason(
            {variable.name: "1" for variable in load_data_access_rules().blocking},
            load_data_access_rules(),
        )
        == load_data_access_rules().blocking[0].reason
    )


def test_blocked_reason_identifies_codex() -> None:
    """Neprázdné ID relace Codexu vrátí důvod jmenující agenta."""
    reason = blocked_reason({"CODEX_THREAD_ID": "019a…"}, load_data_access_rules())
    assert reason is not None
    assert "Codex" in reason


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {variable.name: "" for variable in load_data_access_rules().blocking},
        {"JINE": "1"},
    ],
)
def test_blocked_reason_is_none_without_non_empty_variable(
    environ: dict[str, str],
) -> None:
    """Prázdná nebo chybějící proměnná nic neblokuje."""
    assert blocked_reason(environ, load_data_access_rules()) is None


@pytest.mark.parametrize(
    "variable", load_data_access_rules().blocking, ids=lambda variable: variable.name
)
def test_check_refuses_real_data_in_blocked_environment(
    variable: BlockingVariable,
) -> None:
    """Skutečná data v blokujícím prostředí hláška jmenuje místo, značku i důvod."""
    with pytest.raises(DataAccessError) as refusal:
        check_data_access(
            {variable.name: "1"},
            load_data_access_rules(),
            synthetic=False,
            location=LOCATION,
            marker=MARKER,
        )

    message = str(refusal.value)
    assert LOCATION in message
    assert MARKER in message
    assert variable.reason in message


def test_refusal_is_an_invalid_input() -> None:
    """Odmítnutí přístupu je odmítnutí kroku."""
    assert issubclass(DataAccessError, InvalidInputError)


def test_check_allows_synthetic_data_in_blocked_environment() -> None:
    """Syntetická data projdou i při všech blokujících proměnných."""
    check_data_access(
        {variable.name: "1" for variable in load_data_access_rules().blocking},
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
    "case",
    json.loads(
        (REPOSITORY / "tests/fixtures/data_access_rules_cases.json").read_text(
            encoding="utf-8"
        )
    ),
    ids=itemgetter("name"),
)
def test_shared_rules_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict[str, str | bool]
) -> None:
    """Stejné případy pravidel ověřují pydantic, loader i nezávislý konzument R."""
    content = str(case["text"])
    (tmp_path / "data_access.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "mediparse.infrastructure.data_access_rules.files", lambda _: tmp_path
    )
    if case["valid"]:
        DataAccessRules.model_validate_json(content)
        load_data_access_rules()
    else:
        with pytest.raises(ValidationError):
            DataAccessRules.model_validate_json(content)
        with pytest.raises(InvalidInputError, match=r"data_access\.json"):
            load_data_access_rules()


def test_loader_refuses_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chybějící pravidla se přeloží na doménové odmítnutí."""
    monkeypatch.setattr(
        "mediparse.infrastructure.data_access_rules.files", lambda _: tmp_path
    )
    with pytest.raises(InvalidInputError, match=r"data_access\.json"):
        load_data_access_rules()
