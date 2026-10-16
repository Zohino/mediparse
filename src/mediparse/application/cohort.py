"""Use case materializace kohorty: štítky pětice odlišným kódům a zápis kohorty zdrojem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.cohort import ensure_publishable, label_codes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from contextlib import AbstractContextManager

    from mediparse.domain.cohort import CohortCounts, RawCode
    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.labels import Diagnosis


class SelectedCohort(Protocol):
    """Vybraná kohorta: agregáty k posouzení a zápis přesně těch zpráv, které počítají."""

    @property
    def counts(self) -> CohortCounts:
        """Trychtýř a počty pozitivních zpráv vybrané kohorty."""

    def write(self) -> None:
        """Zapíše vybranou kohortu."""


class CohortSource(Protocol):
    """Tabulky, z nichž kohorta vzniká, a místo, kam se zapisuje."""

    def codes(self) -> Iterable[RawCode]:
        """Odlišné dvojice verze a kódu z tabulky diagnóz."""

    def select(
        self, labels: Mapping[RawCode, Diagnosis]
    ) -> AbstractContextManager[SelectedCohort]:
        """Vybere zprávy kohorty a označí je; výběr žije do opuštění kontextu."""


@dataclass(frozen=True)
class CohortResult:
    """Agregáty zapsané kohorty a počet neplatných dvojic kódu."""

    counts: CohortCounts
    invalid_codes: int


@dataclass(frozen=True)
class CohortMaterialization:
    """Přiřadí diagnózy kódům, ověří pravidlo DUA nad výběrem a nechá zdroj zapsat kohortu."""

    source: CohortSource
    codes: DiagnosisCodes

    def run(self) -> CohortResult:
        """Zapíše kohortu, kterou pravidlo DUA propustí.

        Kohortu, kterou pravidlo odmítne, nezapíše; ``InvalidInputError``
        z pravidla DUA i ze zdroje projde ven.

        Returns:
            Agregáty kohorty a počet neplatných dvojic.
        """
        labelled = label_codes(self.source.codes(), self.codes)
        with self.source.select(labelled.diagnoses) as selected:
            counts = selected.counts
            ensure_publishable(counts)
            selected.write()
        return CohortResult(counts=counts, invalid_codes=labelled.invalid)
