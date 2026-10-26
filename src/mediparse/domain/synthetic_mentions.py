"""Řádky zmínek syntetické zprávy: klíčová slova, zkratky a příznaky ve třech stavech bloku."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.text_mentions import TextMentions

# Zrcadlo SHARE_MIN_OCCURRENCES z eda/R/preprocessing.R. Kapitola Předzpracování zveřejní podíl
# zkratky jen od tohoto počtu výskytů a syntetické zprávy ho musí splnit. Změna se dělá v obou
# souborech, dokud sešit 9 nepřejde do Pythonu (S20h11).
ACRONYM_SHARE_MIN_OCCURRENCES: Final = 1000


class MentionState(Enum):
    """Jak zpráva bloku zmiňuje klíčová slova: tři kategorie sloupců sešitu 8."""

    ASSERTED = auto()
    NEGATED = auto()
    ABSENT = auto()


@dataclass(frozen=True)
class MentionVocabulary:
    """Co zpráva může zmínit a jak ji sešit 8 pozná jako negovanou.

    Attributes:
        phrases: Klíčová slova diagnóz a příznaky, každé je zmínka.
        acronyms: Zkratky diagnóz pětice; sešit 9 počítá jejich výskyty.
        cue: Negační výraz z configu, který zmínku v rámci řádku neguje.
    """

    phrases: tuple[str, ...]
    acronyms: tuple[str, ...]
    cue: str


def mention_lines(
    vocabulary: MentionVocabulary, state: MentionState, acronym_repeats: int
) -> tuple[str, ...]:
    """Vytvoří řádky zmínek pro stav bloku, každou zmínku na vlastním řádku.

    Tvrzený stav opakuje každou zkratku ``acronym_repeats`` krát na jednom řádku, aby výskyty
    dosáhly prahu podílu zkratek. Negovaný stav ji píše jednou: opakování za jedním výrazem by
    vyskočilo z okna negace a stalo se tvrzenou zmínkou.

    Args:
        vocabulary: Fráze, zkratky a negační výraz.
        state: Stav zmínek bloku.
        acronym_repeats: Počet výskytů zkratky na jejím řádku v tvrzeném stavu.

    Returns:
        Řádky zmínek; bez zmínky prázdná n-tice.
    """
    if state is MentionState.ABSENT:
        return ()
    if state is MentionState.NEGATED:
        return tuple(
            f"{vocabulary.cue} {mention}"
            for mention in (*vocabulary.phrases, *vocabulary.acronyms)
        )
    return (
        *vocabulary.phrases,
        *(" ".join([acronym] * acronym_repeats) for acronym in vocabulary.acronyms),
    )


def mention_vocabulary(
    keywords: tuple[str, ...],
    mentions: TextMentions,
    normalization: TextNormalization,
) -> MentionVocabulary:
    """Složí slovník zmínek ze tří configů: klíčová slova, příznaky a zkratky.

    Args:
        keywords: Klíčová slova diagnóz pětice.
        mentions: Negace a příznaky z configu zmínek.
        normalization: Zkratky diagnóz pětice; berou se jejich klíče.

    Returns:
        Slovník, ze kterého ``mention_lines`` skládá řádky.
    """
    return MentionVocabulary(
        phrases=(*keywords, *mentions.symptoms),
        acronyms=tuple(normalization.acronyms),
        # Každá negovaná zmínka stojí na vlastním řádku, takže stačí jediný výraz: další
        # by řádky jen znásobily a okno negace sešitu 8 by se nezměnilo.
        cue=mentions.negation.cues[0],
    )
