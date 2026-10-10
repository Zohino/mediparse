"""Revize zdrojového kódu čtená z gitu na hostiteli."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import SourceRevision

if TYPE_CHECKING:
    from pathlib import Path


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
