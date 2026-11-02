"""Use case vzorkování plánů: z configu a seedu vylosuje plány všech zpráv a předá je k uložení."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.mentions import sample_mentions
from mediparse.domain.note_plan import note_plan
from mediparse.domain.note_structure import sample_structure
from mediparse.domain.synthetic_patients import sample_notes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from random import Random

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig


class PlanStore(Protocol):
    """Úložiště plánů; kam se zapíšou, ví jen adaptér."""

    def save(self, plans: Sequence[NotePlan]) -> None:
        """Uloží plány v zadaném pořadí."""


@dataclass(frozen=True)
class PlanSampling:
    """Vzorkovač plánů; každá fáze losuje z vlastního proudu odvozeného ze seedu."""

    store: PlanStore
    random_source: Callable[[str], Random]

    def run(self, config: SamplerConfig) -> int:
        """Vylosuje labely, strukturu a zmínky všech zpráv a uloží jejich plány.

        Returns:
            Počet uložených plánů.
        """
        notes = sample_notes(
            config.patients, config.labels, self._stream(config, "patients")
        )
        structures = sample_structure(
            notes, config.structure, self._stream(config, "structure")
        )
        mentions = sample_mentions(
            notes, structures, config.mentions, self._stream(config, "mentions")
        )
        plans = tuple(starmap(note_plan, zip(notes, structures, mentions, strict=True)))
        self.store.save(plans)
        return len(plans)

    def _stream(self, config: SamplerConfig, phase: str) -> Random:
        return self.random_source(f"{config.seed}:{phase}")
