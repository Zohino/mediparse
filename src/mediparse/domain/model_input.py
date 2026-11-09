"""Volby vstupu modelu: ablace diagnostických sekcí a úroveň předzpracování textu."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TextPreprocessing(StrEnum):
    """Žebříček předzpracování od původního textu po přepis slovníkem variant."""

    NONE = "none"
    NORMALIZED = "normalized"
    DICTIONARY = "dictionary"


class ModelInput(BaseModel):
    """Sdílené volby vstupu pro řádek matice a config tréninku."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    strip_diagnostic_sections: bool
    text_preprocessing: TextPreprocessing
