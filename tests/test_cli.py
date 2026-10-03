"""Konzolové skripty: chybějící vstupní soubor skončí kódem REFUSED a argumenty validuje doména."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import commit_sha
from mediparse.entrypoints import (
    corpus_audit,
    corpus_consistency,
    corpus_gate,
    corpus_provenance,
    synthetic_plans,
    verbalization_prompt,
)
from mediparse.entrypoints.cli import argument
from mediparse.entrypoints.exit_code import ExitCode
from tests.support import (
    COMMIT,
    REPOSITORY_PLANS,
    REPOSITORY_TABLES,
    REPOSITORY_TEMPLATE,
)

if TYPE_CHECKING:
    from collections.abc import Callable

PROVENANCE = [
    "--model=claude-sonnet-5-5",
    "--claude-code-version=2.1.5",
    "--date=2026-10-10",
    f"--specification-commit={COMMIT}",
]


def _commands(root: Path, missing: Path) -> dict[str, Callable[[], ExitCode]]:
    return {
        "sample-plans": lambda: synthetic_plans.run([
            f"--config={missing}",
            f"--output={root / 'plans.jsonl'}",
            f"--labels={root / 'labels.csv'}",
        ]),
        "verbalization-prompt": lambda: verbalization_prompt.run([
            "90000001-DS-1",
            f"--config={missing}",
            f"--plans={REPOSITORY_PLANS}",
            f"--template={REPOSITORY_TEMPLATE}",
        ]),
        "corpus-check": lambda: corpus_consistency.run([
            f"--corpus={root}",
            f"--plans={REPOSITORY_PLANS}",
            f"--config={missing}",
        ]),
        "corpus-gate": lambda: corpus_gate.run([
            f"--corpus={root}",
            f"--tables={missing}",
        ]),
        "corpus-audit": lambda: corpus_audit.run(
            [
                f"--corpus={root}",
                f"--reference={root / 'discharge.csv.gz'}",
                f"--tables={missing}",
                f"--report={root / 'report.json'}",
                f"--commit={COMMIT}",
            ],
            {},
        ),
        "corpus-provenance": lambda: corpus_provenance.run([
            *PROVENANCE,
            f"--corpus={root}",
            f"--config={missing}",
            f"--plans={REPOSITORY_PLANS}",
            f"--template={REPOSITORY_TEMPLATE}",
            f"--tables={REPOSITORY_TABLES}",
        ]),
    }


@pytest.mark.parametrize("command", list(_commands(Path(), Path())))
def test_missing_input_file_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """Chybějící config nebo tabulky příkaz odmítne hláškou se jménem souboru, ne tracebackem."""
    missing = tmp_path / "missing.json"

    assert _commands(tmp_path, missing)[command]() == ExitCode.REFUSED
    assert str(missing) in capsys.readouterr().err


def test_argument_reports_domain_message() -> None:
    """Chyba doménové validace se v argparse ukáže s hláškou domény."""
    with pytest.raises(argparse.ArgumentTypeError, match="SHA-1"):
        argument(commit_sha)("abc123")
