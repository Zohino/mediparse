"""Use case kontrol shody: každá zpráva korpusu odpovídá svému plánu a korpus jako celek modelu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.corpus_texts import (
    LANGUAGE,
    TRANSLATION_LANGUAGE,
    texts_in_language,
)
from mediparse.domain.corpus_statistics import corpus_violations
from mediparse.domain.note_consistency import note_violations
from mediparse.domain.translation_consistency import (
    marker_mismatches,
    translation_violations,
)

if TYPE_CHECKING:
    from mediparse.application.ports import NoteSource, PlanSource
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig


@dataclass(frozen=True)
class NoteViolations:
    """Zpráva, která neodpovídá plánu, a důvody."""

    note_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConsistencyReport:
    """Výsledek kontrol: zprávy k přegenerování, porušení korpusu a upozornění, která neblokují."""

    notes: tuple[NoteViolations, ...]
    corpus: tuple[str, ...]
    notices: tuple[str, ...]


@dataclass(frozen=True)
class CorpusConsistency:
    """Kontroly shody anglických zpráv s plány, korpusu s modelem a českého překladu s originálem."""

    plans: PlanSource
    corpus: NoteSource

    def run(self, config: SamplerConfig) -> ConsistencyReport:
        """Zkontroluje každou zprávu proti plánu a neprázdný korpus proti modelu.

        Returns:
            Zprávy k přegenerování seřazené podle note_id, porušení korpusu a upozornění
            na neshodu počtu značek ___ v překladu; korpus bez zpráv nic neporušuje,
            částečný korpus je neúplný.
        """
        plans = {plan.note_id: plan for plan in self.plans.load()}
        all_notes = self.corpus.notes()
        texts = texts_in_language(all_notes, LANGUAGE)
        czech = texts_in_language(all_notes, TRANSLATION_LANGUAGE)
        notes = tuple(
            NoteViolations(note_id, reasons)
            for note_id in sorted(texts)
            if (reasons := _reasons(texts[note_id], plans.get(note_id), config))
        )
        corpus = (
            *corpus_violations(texts, plans, config),
            *translation_violations(texts, czech),
        )
        notices = tuple(
            f"cs/{note_id}: jiný počet značek ___ než originál."
            for note_id in marker_mismatches(texts, czech)
        )
        return ConsistencyReport(notes, corpus, notices)


def _reasons(
    text: str, plan: NotePlan | None, config: SamplerConfig
) -> tuple[str, ...]:
    if plan is None:
        return ("Zpráva nemá plán.",)
    return note_violations(text, plan, config.structure, config.mentions)
