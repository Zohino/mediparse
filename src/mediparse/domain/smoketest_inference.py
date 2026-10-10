"""Inference smoketestu: shoda uloženého modelu s tréninkem a výběr ukázky."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.evaluation import Classification, Prediction

TOLERANCE: Final = 1e-9


class ModelMismatchError(InvalidInputError):
    """Uložený model dává jiné predikce než model při tréninku."""


def ensure_predictions_present(expected: Sequence[Prediction]) -> None:
    """Ověří, že predikce z tréninku nejsou prázdné.

    Args:
        expected: Predikce zapsané při tréninku.

    Raises:
        InvalidInputError: Predikce jsou prázdné, není co ověřit.
    """
    if expected:
        return
    msg = "Soubor predikcí je prázdný, není co ověřit."
    raise InvalidInputError(msg)


def ensure_same_predictions(
    expected: Sequence[Prediction], actual: Sequence[Classification]
) -> None:
    """Ověří, že znovu načtený model dává stejné predikce jako při tréninku.

    Skóre se porovnává s tolerancí, protože determinismus platí v mezích, ne bit
    po bitu.

    Args:
        expected: Predikce zapsané při tréninku.
        actual: Klasifikace stejných zpráv uloženým modelem ve stejném pořadí.

    Raises:
        ModelMismatchError: Počty se liší, nebo se u některé zprávy liší label
            či skóre.
    """
    if len(expected) != len(actual):
        msg = f"Neshoda: počet predikcí {len(expected)}, klasifikací {len(actual)}."
        raise ModelMismatchError(msg)
    for item, result in zip(expected, actual, strict=True):
        same_score = math.isclose(
            item.score, result.score, rel_tol=TOLERANCE, abs_tol=TOLERANCE
        )
        if item.predicted == result.predicted and same_score:
            continue
        msg = (
            f"Neshoda u zprávy {item.note_id}: při tréninku label {item.predicted} "
            f"a skóre {item.score}, po načtení modelu label {result.predicted} "
            f"a skóre {result.score}."
        )
        raise ModelMismatchError(msg)


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
