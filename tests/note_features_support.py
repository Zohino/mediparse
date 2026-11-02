"""Zkonstruované zprávy a paměťové porty pro kontrakt měření zpráv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mediparse.application.note_features import ComputeNoteFeatures
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES, REPOSITORY_SECTION_HEADERS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mediparse.application.note_features import NoteFeatureSink
    from mediparse.domain.note_features import AcronymCount, NotePreprocessing
    from mediparse.domain.note_sections import SectionSpan

CONTRACT_NOTES = (
    ("n1", "Name: ___\n\nAllergies:\n___\n\nChief Complaint:\nPt 5mg ___ AF\n"),
    ("n2", "No header and no abbreviation"),
    ("n3", "Allergies:\nnone\nChief Complaint:\n___COPD and DM2, DM; AFib af\n"),
)


@dataclass(frozen=True)
class MemoryTexts:
    """Zdroj dávek zkonstruovaných zpráv."""

    rows: Sequence[tuple[str, str]]

    def batches(self) -> Iterable[Sequence[tuple[str, str]]]:
        """Vrátí jednu zprávu na dávku, aby se ověřilo skládání dávek.

        Returns:
            Jednořádkové dávky.
        """
        return ([row] for row in self.rows)


@dataclass
class MemoryFeatures:
    """Paměťový sink pro očekávání výpočtu."""

    sections: list[SectionSpan] = field(default_factory=list)
    preprocessing: list[NotePreprocessing] = field(default_factory=list)
    acronyms: list[AcronymCount] = field(default_factory=list)

    def write(
        self,
        sections: Sequence[SectionSpan],
        preprocessing: Sequence[NotePreprocessing],
        acronyms: Sequence[AcronymCount],
    ) -> None:
        """Připojí další dávku."""
        self.sections.extend(sections)
        self.preprocessing.extend(preprocessing)
        self.acronyms.extend(acronyms)


def compute(rows: Sequence[tuple[str, str]], sink: NoteFeatureSink) -> int:
    """Složí use case s tréninkovými analyzéry a configy repa.

    Returns:
        Počet zpráv.
    """
    normalization = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES)
    return ComputeNoteFeatures(
        MemoryTexts(rows),
        sink,
        tfidf_vectorizer(None).build_analyzer(),
        tfidf_vectorizer(normalization).build_analyzer(),
    ).run(load_section_lexicon(REPOSITORY_SECTION_HEADERS), normalization)
