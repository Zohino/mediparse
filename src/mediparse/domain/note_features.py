"""Měření analyzéru TF-IDF a výskytů zkratek jedné zprávy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NotePreprocessing:
    """Počty tokenů a seřazené distinct tokeny analyzérů zprávy."""

    note_id: str
    tokens_raw: int
    tokens_preprocessed: int
    vocabulary_raw: tuple[str, ...]
    vocabulary_preprocessed: tuple[str, ...]


@dataclass(frozen=True)
class AcronymCount:
    """Počet výskytů jedné zkratky v normalizované zprávě včetně nuly."""

    note_id: str
    acronym: str
    occurrences: int
