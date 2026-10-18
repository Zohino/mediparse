"""Use case materializace splitu: pacienti zpráv do foldů, uložení pro všechny kroky."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.split import SharedSplit, ensure_enough_patients, strata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import InputTable
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.split import Split, SplitConfig


class PatientSplitter(Protocol):
    """Rozdělení pacientů do foldů."""

    def split(
        self,
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        config: SplitConfig,
    ) -> Split:
        """Rozdělí pacienty zpráv do foldů podle třídy stratifikace každé zprávy."""


class SplitSink(Protocol):
    """Místo, kam se split uloží, aby ho další kroky četly."""

    def write(self, shared: SharedSplit) -> None:
        """Uloží split s jeho parametry."""


@dataclass(frozen=True)
class SplitMaterialization:
    """Spočte split pacientů jednou a uloží ho jako sdílený artefakt."""

    table: InputTable
    splitter: PatientSplitter
    sink: SplitSink

    def run(self, config: SplitConfig) -> SharedSplit:
        """Rozdělí pacienty tabulky do foldů a výsledek zapíše.

        Args:
            config: Seed a počet foldů splitu.

        Returns:
            Zapsaný split s configem.
        """
        notes = self.table.read()
        subjects = [note.subject_id for note in notes]
        ensure_enough_patients(subjects, config.folds)
        split = self.splitter.split(subjects, strata(notes, config.folds), config)
        shared = SharedSplit(config, split)
        self.sink.write(shared)
        return shared
