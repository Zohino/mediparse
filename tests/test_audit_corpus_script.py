"""Kontrola revize receptu auditu nad dočasným repem a neškodným stubem."""

from __future__ import annotations

import os
import shutil

# Test musí spustit skutečný shellový recept; audit nahrazuje lokální stub.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from typing import TYPE_CHECKING

import pytest

from tests.git_support import git_output, init_repo, needs_git
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


@needs_git
@pytest.mark.parametrize(
    ("changed", "allowed"),
    [
        ("results/cohort/manifest.json", True),
        ("src/a.py", False),
        ("resources/synthetic/en/x.txt", True),
    ],
)
def test_audit_recipe_blocks_only_uncommitted_code(
    tmp_path: Path, changed: str, *, allowed: bool
) -> None:
    """Výstup ani auditovaný korpus neblokují recept; necommitnutý kód ano."""
    root = tmp_path / "repo"
    init_repo(root, "syntetický test\n")
    scripts = root / "scripts"
    scripts.mkdir()
    script = scripts / "audit-corpus.sh"
    shutil.copyfile(REPOSITORY / "scripts/audit-corpus.sh", script)
    git_output(root, "add", "scripts")
    git_output(root, "commit", "-q", "-m", "recipe")
    changed_path = root / changed
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed_path.write_text("syntetická změna\n", encoding="utf-8")
    commands = tmp_path / "bin"
    commands.mkdir()
    stub = commands / "mediparse-corpus-audit"
    calls = tmp_path / "calls.txt"
    stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$AUDIT_TEST_CALLS"\n', encoding="utf-8"
    )
    stub.chmod(0o755)
    environ = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environ.update({
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "PIXI_PROJECT_ROOT": str(root),
        "AUDIT_TEST_CALLS": str(calls),
    })

    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["bash", str(script), "discharge", "radiology", "audit.json"],  # ruff: ignore[start-process-with-partial-path]
        cwd=root,
        env=environ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )

    assert result.returncode == (0 if allowed else 1), result.stderr
    if allowed:
        assert calls.read_text(encoding="utf-8").splitlines() == [
            "--reference",
            "discharge",
            "--reference",
            "radiology",
            "--report",
            "audit.json",
            "--commit",
            git_output(root, "rev-parse", "HEAD"),
        ]
    else:
        assert not calls.exists()
        assert "necommitnut" in result.stderr
