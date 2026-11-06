"""Měření analyzéru TF-IDF a výskytů zkratek i variant diagnóz jedné zprávy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediparse.domain.diagnosis import Diagnosis


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


@dataclass(frozen=True)
class VariantCount:
    """Počet výskytů varianty názvu diagnózy v normalizované zprávě včetně nuly."""

    note_id: str
    variant: str
    diagnosis: Diagnosis
    occurrences: int
