"""Zmínky v textu podle EDA: pravidlo negace a příznaky, které sešit 8 hledá."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.mentions import Negation, NonBlank


class TextMentions(BaseModel):
    """Pravidlo negace a příznaky z ``config/text_mentions.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    negation: Negation
    symptoms: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
