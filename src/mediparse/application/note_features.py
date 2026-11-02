"""Výpočet částí zpráv, tokenů TF-IDF a výskytů zkratek po dávkách."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.clinical_text import (
    find_diagnosis_acronyms,
    normalize_clinical_text,
)
from mediparse.domain.note_features import AcronymCount, NotePreprocessing
from mediparse.domain.note_sections import SectionSpan, section_spans

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.section_headers import SectionLexicon


class NoteTexts(Protocol):
    """Zdroj textů celé tabulky v pořadí souboru."""

    def batches(self) -> Iterable[Sequence[tuple[str, str]]]:
        """Vrátí dávky dvojic identifikátor a text."""
        ...


class NoteFeatureSink(Protocol):
    """Zápis měření jedné dávky do tří artefaktů."""

    def write(
        self,
        sections: Sequence[SectionSpan],
        preprocessing: Sequence[NotePreprocessing],
        acronyms: Sequence[AcronymCount],
    ) -> None:
        """Zapíše řádky v daném pořadí."""
        ...


@dataclass(frozen=True)
class ComputeNoteFeatures:
    """Změří všechny zprávy pomocí parseru a analyzérů používaných při tréninku."""

    texts: NoteTexts
    sink: NoteFeatureSink
    raw: Callable[[str], list[str]]
    preprocessed: Callable[[str], list[str]]

    def run(self, lexicon: SectionLexicon, normalization: TextNormalization) -> int:
        """Spočítá a zapíše měření po dávkách bez výběru vstupního korpusu.

        Returns:
            Počet zpracovaných zpráv.
        """
        total = 0
        for batch in self.texts.batches():
            sections: list[SectionSpan] = []
            preprocessing: list[NotePreprocessing] = []
            acronyms: list[AcronymCount] = []
            for note_id, text in batch:
                sections.extend(section_spans(note_id, text, lexicon))
                raw, prepared = self.raw(text), self.preprocessed(text)
                preprocessing.append(
                    NotePreprocessing(
                        note_id,
                        len(raw),
                        len(prepared),
                        tuple(sorted(set(raw))),
                        tuple(sorted(set(prepared))),
                    )
                )
                counts = Counter(
                    find_diagnosis_acronyms(
                        normalize_clinical_text(text),
                        normalization.acronyms,
                    )
                )
                acronyms.extend(
                    AcronymCount(note_id, acronym, counts[acronym])
                    for acronym in sorted(normalization.acronyms)
                )
            self.sink.write(sections, preprocessing, acronyms)
            total += len(batch)
        return total
