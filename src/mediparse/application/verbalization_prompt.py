"""Use case zadání verbalizace: k note_id najde plán a vyrobí z něj zadání pro model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.verbalization import render_prompt

if TYPE_CHECKING:
    from mediparse.application.ports.corpus import PlanSource
    from mediparse.domain.synthetic_plan import SamplerConfig


@dataclass(frozen=True)
class VerbalizationPrompt:
    """Zadání verbalizace jedné zprávy ze šablony instrukcí, jejího plánu a configu."""

    plans: PlanSource

    def run(self, note_id: str, template: str, config: SamplerConfig) -> str:
        """Vyrobí zadání pro zprávu s daným note_id.

        Returns:
            Text zadání.

        Raises:
            InvalidInputError: Plán s tímto note_id neexistuje.
        """
        plan = next(
            (plan for plan in self.plans.load() if plan.note_id == note_id), None
        )
        if plan is None:
            msg = f"Plán pro note_id {note_id} neexistuje."
            raise InvalidInputError(msg)
        return render_prompt(template, plan, config)
