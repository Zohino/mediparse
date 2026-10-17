"""Rozložení vstupů auditu: report nesmí skončit v žádném git repozitáři."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.infrastructure.audit_workspace import LocalWorkspace

if TYPE_CHECKING:
    from pathlib import Path


def _corpus_in_repository(root: Path) -> Path:
    corpus = root / "resources" / "synthetic"
    corpus.mkdir(parents=True)
    (root / ".git").mkdir()
    return corpus


def test_report_in_another_repository_is_inside(outside_repository: Path) -> None:
    """Report v cizím repozitáři je stejné riziko jako v repozitáři korpusu."""
    corpus = _corpus_in_repository(outside_repository / "repo")
    other = outside_repository / "other"
    (other / ".git").mkdir(parents=True)

    workspace = LocalWorkspace(corpus, (), other / "r.json")

    assert workspace.report_in_repository()


def test_report_from_worktree_into_main_checkout_is_inside(
    outside_repository: Path,
) -> None:
    """Z worktree (``.git`` jako soubor) report do hlavního checkoutu neprojde."""
    main = outside_repository / "main"
    (main / ".git").mkdir(parents=True)
    worktree = main / ".worktrees" / "s99-vetev"
    corpus = worktree / "resources" / "synthetic"
    corpus.mkdir(parents=True)
    (worktree / ".git").write_text(
        "gitdir: ../../.git/worktrees/s99\n", encoding="utf-8"
    )

    workspace = LocalWorkspace(corpus, (), main / "r.json")

    assert workspace.corpus_in_repository()
    assert workspace.report_in_repository()


def test_report_outside_every_repository_is_outside(outside_repository: Path) -> None:
    """Report do dosud neexistujícího adresáře mimo repozitáře projde."""
    corpus = _corpus_in_repository(outside_repository / "repo")

    workspace = LocalWorkspace(corpus, (), outside_repository / "out" / "r.json")

    assert not workspace.report_in_repository()
