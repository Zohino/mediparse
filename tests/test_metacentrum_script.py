"""Stráž skriptů, které přenáší data MIMIC mezi laptopem a MetaCentrem."""

from __future__ import annotations

import os
import subprocess

import pytest

from tests.support import BLOCKING_NAMES, REPOSITORY, blocking_reason


@pytest.mark.parametrize(
    "script", ["scripts/metacentrum-submit.sh", "scripts/metacentrum-fetch.sh"]
)
def test_metacentrum_script_blocks_before_ssh(script: str) -> None:
    """Codex zastaví skript dřív, než se spojí s charonem."""
    result = subprocess.run(
        ["bash", str(REPOSITORY / script), "run", "job.sh"],
        cwd=REPOSITORY,
        env={
            **os.environ,
            **dict.fromkeys(BLOCKING_NAMES, ""),
            "CODEX_THREAD_ID": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == 1
    assert blocking_reason("CODEX_THREAD_ID") in result.stderr
