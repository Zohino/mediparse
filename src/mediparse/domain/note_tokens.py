"""Délky tokenů zpráv, nastavení tokenizérů a fakta o dělení překladu."""

from dataclasses import dataclass
from typing import Final, Literal

TOKEN_TABLES: Final = ("discharge", "radiology")


@dataclass(frozen=True)
class NoteTokens:
    """Délky jedné zprávy ve třech tokenizérech."""

    note_id: str
    xlmr: int
    mdeberta: int
    translategemma: int


@dataclass(frozen=True)
class TokenizerInfo:
    """Pevná revize tokenizéru a parametry okna."""

    tokenizer: str
    model: str
    revision: str
    window: int
    role: Literal["encoder", "translation"]
    special_tokens: int
    transformers: str


@dataclass(frozen=True)
class TranslationPrompt:
    """Změřená režie chat šablony a rezerva na překlad."""

    prompt_overhead: int
    split_ratio: float
    sample_limit: int


@dataclass(frozen=True)
class SplitSample:
    """Přesné dělení jedné zprávy deterministického vzorku."""

    note_table: str
    note_id: str
    fits: bool
    parts: int
    unsplittable: bool


@dataclass(frozen=True)
class ParallelTokens:
    """Délky jednoho veřejného syntetického páru bez speciálních tokenů."""

    pair_id: str
    tokenizer: str
    english: int
    czech: int
