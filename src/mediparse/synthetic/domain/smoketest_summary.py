"""Výběr ukázek predikcí pro souhrn smoketestu."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.evaluation import truth_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.evaluation import Prediction
    from mediparse.domain.input_note import InputNote


def demo_examples(
    predictions: Sequence[Prediction], notes: Sequence[InputNote]
) -> tuple[tuple[Prediction, bool], ...]:
    """Vybere první pozitivní a první negativní zprávu podle labelů kohorty a note_id.

    Args:
        predictions: Predikce odložených zpráv.
        notes: Zprávy kohorty s labely diagnóz.

    Returns:
        Dvojice predikce a pravdy, pozitivní před negativní; chybí-li třída,
        jen druhá; prázdný vstup dá prázdnou n-tici.

    Raises:
        InvalidInputError: Některá predikce nemá zprávu v kohortě.
    """
    ordered = sorted(predictions, key=lambda item: item.note_id)
    labeled = tuple(zip(ordered, truth_of(ordered, notes), strict=True))
    firsts = (
        next(
            (
                (item, is_positive)
                for item, is_positive in labeled
                if is_positive is truth
            ),
            None,
        )
        for truth in (True, False)
    )
    return tuple(item for item in firsts if item is not None)
