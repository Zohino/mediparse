"""Konfigurace délky korpusu se stejnými případy pro Python a R."""

from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_corpus_rule import load_input_corpus_rule
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from typing import Any

CASES = json.loads(
    (REPOSITORY / "tests/fixtures/input_corpus_rule_cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", CASES, ids=itemgetter("name"))
def test_shared_rule_cases(tmp_path: Path, case: dict[str, Any]) -> None:
    """Loader přijme jen sdílené platné případy a zbytek odmítne doménovou chybou."""
    path = tmp_path / "rule.json"
    path.write_text(case["text"], encoding="utf-8")
    if case["valid"]:
        load_input_corpus_rule(path)
    else:
        with pytest.raises(InvalidInputError):
            load_input_corpus_rule(path)


def test_repository_rule_matches_protocol() -> None:
    """Dodaný config drží schválený práh sto znaků."""
    expected_min_chars: Final = 100
    rule = load_input_corpus_rule(REPOSITORY / "config/input_corpus.json")
    assert rule.min_chars == expected_min_chars


def test_missing_rule_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící config je doménová chyba."""
    with pytest.raises(InvalidInputError):
        load_input_corpus_rule(tmp_path / "missing.json")


def test_unreadable_rule_is_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chyba oprávnění při čtení configu je doménová chyba."""

    def denied(*_args: object, **_kwargs: object) -> str:
        raise PermissionError

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(InvalidInputError):
        load_input_corpus_rule(tmp_path / "rule.json")
