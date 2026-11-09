"""Use case materializace splitu: rozdělení do částí a uložení pro všechny kroky."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.split import SharedSplit, Split, ensure_enough_patients, strata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports.experiment import InputTable
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import SplitConfig


class PatientPartitioner(Protocol):
    """Stratifikované rozdělení pacientů do disjunktních částí."""

    def partition(
        self,
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        parts: int,
        seed: Seed,
    ) -> tuple[frozenset[SubjectId], ...]:
        """Rozdělí pacienty do daného počtu částí podle stratifikace zpráv."""


class SplitSink(Protocol):
    """Místo, kam se split uloží, aby ho další kroky četly."""

    def write(self, shared: SharedSplit) -> None:
        """Uloží split s jeho parametry."""


@dataclass(frozen=True)
class SplitMaterialization:
    """Spočte split pacientů jednou a uloží ho jako sdílený artefakt."""

    table: InputTable
    partitioner: PatientPartitioner
    sink: SplitSink

    def run(self, config: SplitConfig) -> SharedSplit:
        """Vyčlení vývojovou část vedle foldů evaluace a sdílený split zapíše.

        Args:
            config: Seed a počet foldů splitu.

        Returns:
            Zapsaný split s configem.
        """
        notes = self.table.read()
        subjects = [note.subject_id for note in notes]
        parts = config.folds + 1
        ensure_enough_patients(subjects, parts)
        partition = self.partitioner.partition(
            subjects, strata(notes, parts), parts, config.seed
        )
        shared = SharedSplit(config, Split.from_parts(partition))
        self.sink.write(shared)
        return shared
