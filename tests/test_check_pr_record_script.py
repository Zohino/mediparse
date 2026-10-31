"""Kontrola záznamu PR skriptem nad dočasným souborem s tělem."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import pytest

SCRIPT: Final = Path(__file__).parents[1] / "scripts" / "check-pr-record.sh"
TITLE: Final = "ci(pr): kontroluj ADR (S96)"
USAGE_ERROR: Final = 2


@pytest.mark.parametrize(
    ("title", "body", "expected_code", "messages"),
    [
        pytest.param(TITLE, "Text.\nTask: S96\n", 0, (), id="without-supersession"),
        pytest.param(TITLE, "Ruší: ADR D12\nTask: S96\n", 0, (), id="one-adr"),
        pytest.param(
            TITLE,
            "Ruší: ADR D7\nRuší: ADR D37\nTask: S96\n",
            0,
            (),
            id="multiple-adrs",
        ),
        pytest.param(TITLE, "Ruší: ADR D12\r\nTask: S96\r\n", 0, (), id="crlf"),
        pytest.param(
            TITLE,
            "Ruší: design doc ř. 176\nTask: S96\n",
            1,
            ("Ruší: design doc ř. 176", "Ruší: ADR D", "design doc je archivovaný"),
            id="archived-design-doc",
        ),
        pytest.param(
            TITLE,
            "Ruší: ADR 12\nTask: S96\n",
            1,
            ("Ruší: ADR 12", "Ruší: ADR D"),
            id="missing-d",
        ),
        pytest.param(
            TITLE,
            "Ruší: ADR D12, D13\nTask: S96\n",
            1,
            ("Ruší: ADR D12, D13", "Ruší: ADR D"),
            id="comma-list",
        ),
        pytest.param(TITLE, "Ruší: ADR D12 \nTask: S96\n", 1, (), id="trailing-space"),
        pytest.param(TITLE, "Text.\n", 1, ("Task: SNN",), id="missing-task"),
        pytest.param(
            "oprav to",
            "Ruší: design doc ř. 33\n",
            1,
            ("Titulek PR nemá tvar", "Task: SNN", "Ruší: design doc ř. 33"),
            id="all-three-errors",
        ),
        pytest.param(
            TITLE,
            "Ruší: ADR 12\nRuší: design doc ř. 176\nTask: S96\n",
            1,
            ("Ruší: ADR 12", "Ruší: design doc ř. 176"),
            id="all-invalid-lines",
        ),
        pytest.param(TITLE, "Task: S96\nRuší:", 1, ("Ruší: ADR D",), id="bare-key"),
        pytest.param(
            TITLE,
            "Text zmiňuje Ruší: design doc ř. 176\nTask: S96\n",
            0,
            (),
            id="prefix-only",
        ),
        pytest.param(
            TITLE,
            "Task: S96\nRuší: ADR 12",
            1,
            ("Ruší: ADR 12",),
            id="no-final-newline",
        ),
    ],
)
def test_pr_record_validation(
    tmp_path: Path,
    title: str,
    body: str,
    expected_code: int,
    messages: tuple[str, ...],
) -> None:
    """Záznam vrací předepsaný kód a vypíše všechny chyby včetně jejich řádků."""
    body_path = tmp_path / "body.md"
    body_path.write_bytes(body.encode("utf-8"))

    result = subprocess.run(
        ["bash", str(SCRIPT), title, str(body_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == expected_code, result.stderr
    for message in messages:
        assert message in result.stderr
    if expected_code == 0:
        assert not result.stderr


def test_missing_body_argument_is_usage_error() -> None:
    """Jediný argument znamená chybu volání s kódem 2."""
    result = subprocess.run(
        ["bash", str(SCRIPT), TITLE],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == USAGE_ERROR
    assert "Použití:" in result.stderr
