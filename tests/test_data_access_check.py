"""Konzolová stráž skutečných dat MIMIC pro shellové kroky."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.entrypoints.data_access_check import run
from mediparse.entrypoints.exit_code import ExitCode
from tests.support import BLOCKING_NAMES, blocking_reason

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", BLOCKING_NAMES)
def test_blocking_variable_is_logged_and_blocks(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Neprázdná proměnná blokuje s důvodem ze sdílených pravidel."""
    assert run({name: "1"}) == ExitCode.BLOCKED
    assert blocking_reason(name) in caplog.text
    assert caplog.records[-1].levelname == "ERROR"


@pytest.mark.parametrize("environ", [{}, dict.fromkeys(BLOCKING_NAMES, "")])
def test_unblocked_environment_is_allowed(environ: dict[str, str]) -> None:
    """Chybějící i prázdné blokující proměnné dovolí běh."""
    assert run(environ) == ExitCode.OK


def test_invalid_rules_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Neplatná pravidla odmítnou běh i v neblokujícím prostředí."""
    (tmp_path / "data_access.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "mediparse.infrastructure.data_access_rules.files", lambda _: tmp_path
    )

    assert run({}) == ExitCode.REFUSED
    assert "data_access.json" in caplog.text
