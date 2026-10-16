"""Rozložení vstupů auditu na lokálním disku: reference, repozitář korpusu a cíl reportu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LocalWorkspace:
    """Vstupy auditu tak, jak je zadal autor na příkazové řádce."""

    corpus: Path
    references: tuple[Path, ...]
    report: Path

    def missing_references(self) -> list[str]:
        """Referenční soubory, které neexistují.

        Returns:
            Cesty chybějících souborů.
        """
        return [str(path) for path in self.references if not path.is_file()]

    def corpus_in_repository(self) -> bool:
        """Zda korpus leží v git repozitáři.

        Returns:
            True, když nad korpusem existuje adresář s ``.git``.
        """
        return _inside_repository(self.corpus)

    def report_in_repository(self) -> bool:
        """Zda report míří dovnitř libovolného git repozitáře.

        Nestačí repozitář korpusu: z worktree by report prošel do hlavního
        checkoutu, který leží nad ním.

        Returns:
            True, když nad reportem existuje adresář s ``.git``.
        """
        return _inside_repository(self.report)


def _inside_repository(path: Path) -> bool:
    resolved = path.resolve()
    return any((folder / ".git").exists() for folder in (resolved, *resolved.parents))
