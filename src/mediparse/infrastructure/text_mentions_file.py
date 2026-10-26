"""Načtení configu zmínek EDA: pravidlo negace a příznaky."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.mentions import Negation, TextMentions
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


class _Symptoms(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    keywords_en: tuple[str, ...] = Field(min_length=1)


class _File(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    negation: Negation
    symptoms: _Symptoms


def load_text_mentions(path: Path) -> TextMentions:
    """Přečte negační výrazy, okno negace a příznaky z ``config/text_mentions.json``.

    Returns:
        Zmínky, které prošly schématem.
    """
    return parse_file(path, _parse)


def _parse(text: str) -> TextMentions:
    parsed = _File.model_validate_json(text)
    return TextMentions(negation=parsed.negation, symptoms=parsed.symptoms.keywords_en)
