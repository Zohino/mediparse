"""Výpočet částí zpráv, tokenů TF-IDF a výskytů zkratek po dávkách."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.clinical_text import (
    find_diagnosis_acronyms,
    find_diagnosis_variants,
    normalize_clinical_text,
)
from mediparse.domain.note_features import AcronymCount, NotePreprocessing, VariantCount
from mediparse.domain.note_sections import SectionSpan, section_spans

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mediparse.application.ports.corpus import NoteTexts
    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.section_headers import SectionLexicon


class NoteFeatureSink(Protocol):
    """Zápis měření jedné dávky do čtyř artefaktů."""

    def write(
        self,
        sections: Sequence[SectionSpan],
        preprocessing: Sequence[NotePreprocessing],
        acronyms: Sequence[AcronymCount],
        variants: Sequence[VariantCount],
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

    def run(
        self,
        lexicon: SectionLexicon,
        normalization: TextNormalization,
        variants: Mapping[str, Diagnosis],
    ) -> int:
        """Spočítá a zapíše měření po dávkách bez výběru vstupního korpusu.

        Returns:
            Počet zpracovaných zpráv.
        """
        total = 0
        for batch in self.texts.batches():
            sections: list[SectionSpan] = []
            preprocessing: list[NotePreprocessing] = []
            acronyms: list[AcronymCount] = []
            variant_rows: list[VariantCount] = []
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
                normalized = normalize_clinical_text(text)
                counts = Counter(
                    find_diagnosis_acronyms(
                        normalized,
                        normalization.acronyms,
                    )
                )
                acronyms.extend(
                    AcronymCount(note_id, acronym, counts[acronym])
                    for acronym in sorted(normalization.acronyms)
                )
                variant_counts = Counter(find_diagnosis_variants(normalized, variants))
                variant_rows.extend(
                    VariantCount(
                        note_id, variant, variants[variant], variant_counts[variant]
                    )
                    for variant in sorted(variants)
                )
            self.sink.write(sections, preprocessing, acronyms, variant_rows)
            total += len(batch)
        return total
