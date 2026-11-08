"""Stráž shellové kontroly otisků tabulek MIMIC."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

from tests.support import BLOCKING_NAMES, REPOSITORY, blocking_reason

if TYPE_CHECKING:
    from pathlib import Path


def test_verify_mimic_blocks_before_reading_tables(tmp_path: Path) -> None:
    """Codex zastaví kontrolu před čtením tabulek z neexistujícího adresáře."""
    result = subprocess.run(
        [
            "bash",
            str(REPOSITORY / "scripts/verify-mimic.sh"),
            str(tmp_path / "missing"),
        ],
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
    assert "FAILED" not in result.stdout + result.stderr
    assert "No such file" not in result.stderr
