"""Use case kontrol shody: každá zpráva korpusu odpovídá svému plánu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from mediparse.domain.note_consistency import note_violations

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig

LANGUAGE: Final = "en"
_SUFFIX: Final = ".txt"


class PlanSource(Protocol):
    """Plány zpráv korpusu."""

    def load(self) -> tuple[NotePlan, ...]:
        """Načte plány.

        Returns:
            Plány v pořadí zdroje.
        """


class NoteSource(Protocol):
    """Texty zpráv korpusu."""

    def notes(self) -> Mapping[str, str]:
        """Texty zpráv.

        Returns:
            Slovník relativní cesta ``<jazyk>/<note_id>.txt`` → text zprávy.
        """


@dataclass(frozen=True)
class NoteViolations:
    """Zpráva, která neodpovídá plánu, a důvody."""

    note_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CorpusConsistency:
    """Kontroly shody anglických zpráv s plány; plán bez zprávy se přeskočí."""

    plans: PlanSource
    corpus: NoteSource

    def run(self, config: SamplerConfig) -> tuple[NoteViolations, ...]:
        """Zkontroluje každou zprávu, která v korpusu je, proti jejímu plánu.

        Returns:
            Zprávy k přegenerování seřazené podle note_id; prázdný výsledek znamená shodu.
        """
        plans = {plan.note_id: plan for plan in self.plans.load()}
        texts = {
            path.removeprefix(f"{LANGUAGE}/").removesuffix(_SUFFIX): text
            for path, text in self.corpus.notes().items()
            if path.startswith(f"{LANGUAGE}/")
        }
        results = (
            NoteViolations(
                note_id, _reasons(texts[note_id], plans.get(note_id), config)
            )
            for note_id in sorted(texts)
        )
        return tuple(result for result in results if result.reasons)


def _reasons(
    text: str, plan: NotePlan | None, config: SamplerConfig
) -> tuple[str, ...]:
    if plan is None:
        return ("Zpráva nemá plán.",)
    return note_violations(text, plan, config.structure, config.mentions)
