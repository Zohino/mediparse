"""Čtení revize zdroje z gitu: commit, špinavý strom a odmítnutí cizího adresáře."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import SourceRevision
from mediparse.infrastructure.source_revision import (
    EnvironmentRevision,
    GitRevision,
    revision_source,
)

if TYPE_CHECKING:
    from pathlib import Path

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git chybí")


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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Čerstvé repo s jedním commitem a ignorovaným vzorem ``*.log``.

    Returns:
        Kořen repa.
    """
    root = tmp_path / "repo"
    init_repo(root, "a\n")
    return root


@needs_git
def test_clean_repo_returns_head(repo: Path) -> None:
    """Čistý strom vrátí SHA HEAD a dirty False."""
    revision = GitRevision(repo).read()

    assert revision.commit == git_output(repo, "rev-parse", "HEAD")
    assert revision.dirty is False


@needs_git
def test_modified_tracked_file_is_dirty(repo: Path) -> None:
    """Úprava trackovaného souboru je dirty."""
    (repo / "tracked.txt").write_text("b\n", encoding="utf-8")

    assert GitRevision(repo).read().dirty is True


@needs_git
def test_untracked_file_is_dirty(repo: Path) -> None:
    """Nový netrackovaný soubor je dirty."""
    (repo / "new.py").write_text("x\n", encoding="utf-8")

    assert GitRevision(repo).read().dirty is True


@needs_git
def test_ignored_file_is_not_dirty(repo: Path) -> None:
    """Gitignorovaný soubor dirty nezmění."""
    (repo / "run.log").write_text("x\n", encoding="utf-8")

    assert GitRevision(repo).read().dirty is False


@needs_git
def test_git_environment_does_not_override_root(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proměnné GIT_* z prostředí nepřebijí zadaný kořen repa."""
    other = tmp_path / "other"
    init_repo(other, "other\n")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))

    assert GitRevision(repo).read().commit == git_output(repo, "rev-parse", "HEAD")


def test_directory_outside_repo_is_refused(tmp_path: Path) -> None:
    """Adresář mimo repozitář (nebo chybějící git) skončí InvalidInputError."""
    with pytest.raises(InvalidInputError):
        GitRevision(tmp_path).read()


SHA = "a" * 40


def test_environment_revision_reads_commit_and_dirty() -> None:
    """Commit a dirty true/false z prostředí dají revizi zdroje."""
    clean = EnvironmentRevision({"MEDIPARSE_COMMIT": SHA, "MEDIPARSE_DIRTY": "false"})
    dirty = EnvironmentRevision({"MEDIPARSE_COMMIT": SHA, "MEDIPARSE_DIRTY": "true"})

    assert clean.read() == SourceRevision(commit=SHA, dirty=False)
    assert dirty.read() == SourceRevision(commit=SHA, dirty=True)


@pytest.mark.parametrize(
    "environ",
    [
        {"MEDIPARSE_DIRTY": "false"},
        {"MEDIPARSE_COMMIT": "", "MEDIPARSE_DIRTY": "false"},
        {"MEDIPARSE_COMMIT": "xyz", "MEDIPARSE_DIRTY": "false"},
        {"MEDIPARSE_COMMIT": SHA},
        {"MEDIPARSE_COMMIT": SHA, "MEDIPARSE_DIRTY": ""},
        {"MEDIPARSE_COMMIT": SHA, "MEDIPARSE_DIRTY": "True"},
        {"MEDIPARSE_COMMIT": SHA, "MEDIPARSE_DIRTY": "1"},
    ],
)
def test_environment_revision_rejects_invalid(environ: dict[str, str]) -> None:
    """Chybějící či prázdný commit, neplatné SHA a dirty mimo true/false jsou odmítnuty."""
    with pytest.raises(InvalidInputError, match="smoketest"):
        EnvironmentRevision(environ).read()


def test_revision_source_reads_environment_when_commit_variable_exists(
    tmp_path: Path,
) -> None:
    """Proměnná MEDIPARSE_COMMIT vybere revizi z prostředí, ne z gitu."""
    environ = {"MEDIPARSE_COMMIT": "a" * 40, "MEDIPARSE_DIRTY": "true"}

    source = revision_source(environ, tmp_path)

    assert source() == SourceRevision(commit="a" * 40, dirty=True)


def test_revision_source_does_not_fall_back_when_commit_variable_is_empty(
    tmp_path: Path,
) -> None:
    """Prázdná MEDIPARSE_COMMIT (image bez build-argu) neskončí u gitu, ale hláškou o skriptu."""
    environ = {"MEDIPARSE_COMMIT": "", "MEDIPARSE_DIRTY": ""}

    source = revision_source(environ, tmp_path)

    with pytest.raises(InvalidInputError, match=r"smoketest\.sh"):
        source()


@needs_git
def test_revision_source_reads_head_of_root_without_commit_variable(
    tmp_path: Path,
) -> None:
    """Bez proměnné MEDIPARSE_COMMIT vrátí zdroj HEAD zadaného kořene."""
    root = tmp_path / "repo"
    init_repo(root, "a\n")

    source = revision_source({}, root)

    assert source().commit == git_output(root, "rev-parse", "HEAD")


def test_revision_source_reads_git_without_commit_variable(tmp_path: Path) -> None:
    """Bez proměnné MEDIPARSE_COMMIT se revize čte z gitu v daném kořeni."""
    source = revision_source({"MEDIPARSE_DIRTY": "false"}, tmp_path)

    with pytest.raises(InvalidInputError, match="z gitu"):
        source()
