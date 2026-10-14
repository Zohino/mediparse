"""Use case materializace kohorty: štítky pětice odlišným kódům a zápis kohorty zdrojem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.cohort import label_codes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mediparse.domain.cohort import CohortCounts, RawCode
    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.labels import Diagnosis


class CohortSource(Protocol):
    """Tabulky, z nichž kohorta vzniká, a místo, kam se zapisuje."""

    def codes(self) -> Iterable[RawCode]:
        """Odlišné dvojice verze a kódu z tabulky diagnóz."""

    def write(self, labels: Mapping[RawCode, Diagnosis]) -> CohortCounts:
        """Vybere zprávy kohorty, označí je a zapíše; vrátí agregáty."""


@dataclass(frozen=True)
class CohortResult:
    """Agregáty zapsané kohorty a počet neplatných dvojic kódu."""

    counts: CohortCounts
    invalid_codes: int


@dataclass(frozen=True)
class CohortMaterialization:
    """Přiřadí diagnózy kódům podle domény a nechá zdroj zapsat kohortu."""

    source: CohortSource
    codes: DiagnosisCodes

    def run(self) -> CohortResult:
        """Zapíše kohortu.

        Returns:
            Agregáty kohorty a počet neplatných dvojic.
        """
        labelled = label_codes(self.source.codes(), self.codes)
        counts = self.source.write(labelled.diagnoses)
        return CohortResult(counts=counts, invalid_codes=labelled.invalid)
