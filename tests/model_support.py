"""Sdílená data testů modelů: predikce a config tréninku ve tvaru doménových typů."""

from __future__ import annotations

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Prediction


def prediction(
    note_id: str,
    *,
    truth: bool,
    score: float,
    diagnosis: Diagnosis = Diagnosis.DIABETES,
    fold: int = 0,
    row_id: str = "row",
) -> Prediction:
    """Predikce, jejíž příznak ``predicted`` plyne ze znaménka skóre.

    Returns:
        Predikce pacienta 1, kladná při ``score > 0``.
    """
    return Prediction(
        row_id,
        fold,
        note_id,
        1,
        diagnosis,
        truth=truth,
        predicted=score > 0,
        score=score,
    )


def training_config(**overrides: object) -> dict[str, object]:
    """Platný config tréninku ve tvaru JSON; test změní jen pole, o kterém je.

    Returns:
        Slovník s klíči ``row_id``, ``diagnosis``, ``seed``, ``folds``
        a ``regularization``, přepsanými hodnotami z ``overrides``.
    """
    return {
        "row_id": "row",
        "diagnosis": "diabetes",
        "seed": 0,
        "folds": 2,
        "regularization": 1.0,
    } | overrides
