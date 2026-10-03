"""Use case kontrol shody: každá zpráva korpusu odpovídá svému plánu a korpus jako celek modelu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.corpus_statistics import corpus_violations
from mediparse.domain.note_consistency import note_violations

if TYPE_CHECKING:
    from mediparse.application.ports import NoteSource, PlanSource
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig

LANGUAGE: Final = "en"
_SUFFIX: Final = ".txt"


@dataclass(frozen=True)
class NoteViolations:
    """Zpráva, která neodpovídá plánu, a důvody."""

    note_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConsistencyReport:
    """Výsledek kontrol: zprávy k přegenerování a porušení korpusu jako celku."""

    notes: tuple[NoteViolations, ...]
    corpus: tuple[str, ...]


@dataclass(frozen=True)
class CorpusConsistency:
    """Kontroly shody anglických zpráv s plány a korpusu s modelem."""

    plans: PlanSource
    corpus: NoteSource

    def run(self, config: SamplerConfig) -> ConsistencyReport:
        """Zkontroluje každou zprávu proti plánu a neprázdný korpus proti modelu.

        Returns:
            Zprávy k přegenerování seřazené podle note_id a porušení korpusu; korpus bez
            zpráv nic neporušuje, částečný korpus je neúplný.
        """
        plans = {plan.note_id: plan for plan in self.plans.load()}
        prefix = f"{LANGUAGE}/"
        texts = {
            path.removeprefix(prefix).removesuffix(_SUFFIX): text
            for path, text in self.corpus.notes().items()
            if path.startswith(prefix)
        }
        notes = tuple(
            NoteViolations(note_id, reasons)
            for note_id in sorted(texts)
            if (reasons := _reasons(texts[note_id], plans.get(note_id), config))
        )
        return ConsistencyReport(notes, corpus_violations(texts, plans, config))


def _reasons(
    text: str, plan: NotePlan | None, config: SamplerConfig
) -> tuple[str, ...]:
    if plan is None:
        return ("Zpráva nemá plán.",)
    return note_violations(text, plan, config.structure, config.mentions)
