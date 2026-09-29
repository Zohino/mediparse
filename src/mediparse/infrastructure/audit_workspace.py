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
        return self._repository() is not None

    def report_in_repository(self) -> bool:
        """Zda report míří dovnitř repozitáře korpusu.

        Returns:
            True, když by report skončil v repozitáři korpusu.
        """
        repository = self._repository()
        return repository is not None and self.report.resolve().is_relative_to(
            repository
        )

    def _repository(self) -> Path | None:
        resolved = self.corpus.resolve()
        return next(
            (
                folder
                for folder in (resolved, *resolved.parents)
                if (folder / ".git").exists()
            ),
            None,
        )
