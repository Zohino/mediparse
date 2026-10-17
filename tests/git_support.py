"""Pomocníci testů nad skutečným gitem: značka přeskočení, spuštění a zakládání repa."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from pathlib import Path

needs_git: Final = pytest.mark.skipif(shutil.which("git") is None, reason="git chybí")


def git_output(root: Path, *args: str) -> str:
    """Spustí git v ``root`` bez proměnných GIT_*.

    Returns:
        Oříznutý stdout.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return done.stdout.strip()


def init_repo(root: Path, content: str) -> None:
    """Založí repo s jedním commitem a ignorovaným vzorem ``*.log``."""
    root.mkdir()
    git_output(root, "init", "-q")
    git_output(root, "config", "user.name", "Test")
    git_output(root, "config", "user.email", "test@example.com")
    git_output(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (root / "tracked.txt").write_text(content, encoding="utf-8")
    git_output(root, "add", "-A")
    git_output(root, "commit", "-q", "-m", "init")
