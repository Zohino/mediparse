"""Řádky zmínek syntetické zprávy podle stavu bloku: tvrzená, negovaná, bez zmínky."""

from __future__ import annotations

import re
from typing import Final

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.mentions import Negation
from mediparse.domain.synthetic_mentions import (
    MentionState,
    MentionVocabulary,
    mention_lines,
    mention_vocabulary,
)
from mediparse.domain.text_mentions import TextMentions

CUE: Final = "no"
REPEATS: Final = 4
VOCABULARY: Final = MentionVocabulary(
    phrases=("diabetes", "chest pain", "end-stage renal disease"),
    acronyms=("DM", "COPD"),
    cue=CUE,
)


def test_asserted_lines_are_phrases_then_one_repeated_line_per_acronym() -> None:
    """Tvrzený stav: každá fráze na vlastním řádku, pak pro každou zkratku řádek s opakovanými výskyty."""
    lines = mention_lines(VOCABULARY, MentionState.ASSERTED, REPEATS)

    assert lines == (
        "diabetes",
        "chest pain",
        "end-stage renal disease",
        "DM DM DM DM",
        "COPD COPD COPD COPD",
    )


def test_negated_lines_put_the_cue_before_every_phrase_and_acronym_once() -> None:
    """Negovaný stav: výraz a fráze na řádku, každá zkratka právě jednou bez ohledu na opakování."""
    lines = mention_lines(VOCABULARY, MentionState.NEGATED, REPEATS)

    assert lines == (
        "no diabetes",
        "no chest pain",
        "no end-stage renal disease",
        "no DM",
        "no COPD",
    )


def test_absent_state_has_no_lines() -> None:
    """Bez zmínky není žádný řádek."""
    assert mention_lines(VOCABULARY, MentionState.ABSENT, REPEATS) == ()


def test_asserted_lines_never_contain_the_cue_as_a_word() -> None:
    """Tvrzený řádek nemá negační výraz jako celé slovo, jinak by ho sešit 8 počítal jako negovaný."""
    lines = mention_lines(VOCABULARY, MentionState.ASSERTED, REPEATS)

    assert not [line for line in lines if re.search(rf"\b{CUE}\b", line.lower())]


def test_vocabulary_joins_keywords_symptoms_acronyms_and_first_cue() -> None:
    """Fráze jsou klíčová slova a příznaky v tomto pořadí, zkratky klíče slovníku a výraz první negační výraz."""
    mentions = TextMentions(
        negation=Negation(cues=("no", "denies"), window_chars=40),
        symptoms=("fever", "cough"),
    )
    normalization = TextNormalization(acronyms={"DM": "diabetes", "DM2": "diabetes"})

    vocabulary = mention_vocabulary(("diabetes", "diabetic"), mentions, normalization)

    assert vocabulary == MentionVocabulary(
        phrases=("diabetes", "diabetic", "fever", "cough"),
        acronyms=("DM", "DM2"),
        cue="no",
    )
