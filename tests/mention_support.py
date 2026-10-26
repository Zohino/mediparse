"""Slovník zmínek z configů repozitáře pro testy syntetických tabulek."""

from __future__ import annotations

from typing import Final

from mediparse.domain.synthetic_mentions import MentionVocabulary
from mediparse.infrastructure.diagnosis_candidates_file import (
    load_diagnosis_keywords,
    load_text_normalization,
)
from mediparse.infrastructure.text_mentions_file import load_text_mentions
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES, REPOSITORY_TEXT_MENTIONS

TEXT_MENTIONS: Final = load_text_mentions(REPOSITORY_TEXT_MENTIONS)
ACRONYMS: Final = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES).acronyms
REPOSITORY_VOCABULARY: Final = MentionVocabulary(
    phrases=(
        *load_diagnosis_keywords(REPOSITORY_DIAGNOSIS_CANDIDATES),
        *TEXT_MENTIONS.symptoms,
    ),
    acronyms=tuple(ACRONYMS),
    cue=TEXT_MENTIONS.negation.cues[0],
)
