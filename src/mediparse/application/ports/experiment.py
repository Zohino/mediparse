"""Porty experimentu ML: vstupní tabulka, klasifikátor a predikce."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.evaluation import Classification, Prediction
    from mediparse.domain.input_note import InputNote


class InputTable(Protocol):
    """Vstupní tabulka zpráv s labely."""

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky v pořadí tabulky."""


class TextClassifier(Protocol):
    """Natrénovaný binární klasifikátor textu."""

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        """Určí label a rozhodovací skóre pro každý text."""


class PredictionSource(Protocol):
    """Zdroj predikcí testovacích zpráv."""

    def read(self) -> tuple[Prediction, ...]:
        """Přečte predikce v pořadí zdroje."""


class PredictionSink(Protocol):
    """Výstup predikcí testovacích zpráv."""

    def write(self, predictions: Sequence[Prediction]) -> None:
        """Zapíše predikce."""
