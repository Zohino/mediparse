"""Revize zdrojového kódu čtená z gitu na hostiteli nebo z prostředí kontejneru."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import SourceRevision

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

COMMIT_VARIABLE: Final = "MEDIPARSE_COMMIT"
DIRTY_VARIABLE: Final = "MEDIPARSE_DIRTY"
DIRTY_VALUES: Final = {"true": True, "false": False}


@dataclass(frozen=True)
class GitRevision:
    """Commit a stav pracovního stromu repozitáře v adresáři ``root``."""

    root: Path

    def read(self) -> SourceRevision:
        """Přečte HEAD a zjistí, zda strom obsahuje změny včetně netrackovaných.

        Returns:
            Revize zdroje.
        """
        commit = self._git("rev-parse", "HEAD")
        dirty = bool(self._git("status", "--porcelain"))
        return SourceRevision(commit=commit, dirty=dirty)

    def _git(self, *args: str) -> str:
        try:
            done = subprocess.run(
                ["git", "-C", str(self.root), *args],
                check=True,
                capture_output=True,
                text=True,
                env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
            )
        except (OSError, subprocess.CalledProcessError) as error:
            msg = f"Revizi zdroje v {self.root} nelze přečíst z gitu: {error}"
            raise InvalidInputError(msg) from error
        return done.stdout.strip()


@dataclass(frozen=True)
class EnvironmentRevision:
    """Commit a stav stromu předané do image jako proměnné prostředí při buildu."""

    environ: Mapping[str, str]

    def read(self) -> SourceRevision:
        """Přečte commit a příznak rozpracovaného stromu z prostředí.

        Returns:
            Revize zdroje.

        Raises:
            InvalidInputError: Commit chybí nebo není SHA, nebo dirty není true/false.
        """
        commit = self.environ.get(COMMIT_VARIABLE, "")
        dirty = self.environ.get(DIRTY_VARIABLE, "")
        if dirty not in DIRTY_VALUES:
            reason = f"{DIRTY_VARIABLE} musí být true nebo false: {dirty!r}"
            raise InvalidInputError(_message(reason))
        try:
            return SourceRevision(commit=commit, dirty=DIRTY_VALUES[dirty])
        except ValidationError as error:
            reason = f"{COMMIT_VARIABLE} není commit SHA: {commit!r}"
            raise InvalidInputError(_message(reason)) from error


def revision_source(
    environ: Mapping[str, str], root: Path
) -> Callable[[], SourceRevision]:
    """Vybere zdroj revize: prostředí image, má-li MEDIPARSE_COMMIT, jinak git.

    Args:
        environ: Proměnné prostředí procesu.
        root: Kořen repozitáře pro čtení z gitu.

    Returns:
        Funkce, která přečte revizi zdroje.
    """
    if COMMIT_VARIABLE in environ:
        return EnvironmentRevision(environ).read
    return GitRevision(root).read


def _message(reason: str) -> str:
    return (
        f"Revizi zdroje nelze přečíst z prostředí: {reason}. "
        "Image se sestavuje přes smoketest.sh nebo smoketest.ps1."
    )
