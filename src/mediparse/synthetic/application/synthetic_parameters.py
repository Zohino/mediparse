"""Use case převzetí parametrů pětice z exportu sešitu 8 do configu vzorkovače."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.synthetic_parameters import required_notes

if TYPE_CHECKING:
    from mediparse.domain.synthetic_parameters import SyntheticParameters


class ParameterExport(Protocol):
    """Export parametrů pětice; odkud pochází, ví jen adaptér."""

    def load(self) -> SyntheticParameters:
        """Přečte export.

        Raises:
            InvalidInputError: Export chybí, nebo neodpovídá schématu.
        """


class SamplerConfigStore(Protocol):
    """Config vzorkovače; jak se parametry slijí do jeho souboru, ví jen adaptér."""

    def apply(self, parameters: SyntheticParameters, notes: int) -> None:
        """Zapíše parametry a počet zpráv, jen když výsledný config projde validací.

        Raises:
            InvalidInputError: Config nejde načíst nebo výsledek není platný;
                nic se nezapíše.
        """


@dataclass(frozen=True)
class ParameterTransfer:
    """Převezme export do configu a odvodí počet zpráv podle specifikace korpusu."""

    export: ParameterExport
    config: SamplerConfigStore

    def run(self) -> int:
        """Načte export a zapíše ho do configu.

        Odmítnutý export nebo config projde jako ``InvalidInputError`` a config
        zůstane beze změny.

        Returns:
            Počet zpráv zapsaný do configu.
        """
        parameters = self.export.load()
        notes = required_notes(parameters.prevalence)
        self.config.apply(parameters, notes)
        return notes
