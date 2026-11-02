"""Výběr ukázek predikcí pro souhrn smoketestu."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.evaluation import Prediction


def demo_examples(predictions: Sequence[Prediction]) -> tuple[Prediction, ...]:
    """Vybere první pozitivní a první negativní zprávu podle pravdy a note_id.

    Args:
        predictions: Predikce odložených zpráv.

    Returns:
        Pozitivní před negativní; chybí-li třída, jen druhá; jinak prázdná n-tice.
    """
    ordered = sorted(predictions, key=lambda item: item.note_id)
    firsts = (
        next((item for item in ordered if item.truth is truth), None)
        for truth in (True, False)
    )
    return tuple(item for item in firsts if item is not None)
