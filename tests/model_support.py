"""Sdílená data testů modelů: predikce a config tréninku ve tvaru doménových typů."""

from __future__ import annotations

from typing import Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language

SEPARABLE_VOCABULARY: Final = (
    "diabetes insulin glucose metformin hyperglycemia",
    "fracture cast orthopedic surgery splint",
)


def separable_notes(
    patients: int, vocabulary: tuple[str, str] = SEPARABLE_VOCABULARY
) -> list[InputNote]:
    """Zprávy pacientů střídavě s diabetem a bez něj, pacient má 1 až 7 zpráv nepravidelně.

    Args:
        patients: Počet pacientů; sudí mají diabetes.
        vocabulary: Slova zpráv pozitivních a negativních pacientů; společná
            slova z nich dělají neseparovatelný korpus.

    Returns:
        Zprávy po pacientech v pořadí pacientů.
    """
    notes = []
    for patient in range(patients):
        positive = patient % 2 == 0
        labels = frozenset({Diagnosis.DIABETES} if positive else ())
        notes.extend(
            InputNote(
                f"{patient}-DS-{visit}",
                patient,
                Language.EN,
                f"{vocabulary[0] if positive else vocabulary[1]} visit {visit}",
                labels,
            )
            for visit in range(1 + (patient * patient * 3 + patient) % 7)
        )
    return notes


def prediction(
    note_id: str,
    *,
    truth: bool,
    score: float,
    diagnosis: Diagnosis = Diagnosis.DIABETES,
    fold: int = 0,
    row_id: str = "row",
    subject_id: int = 1,
) -> Prediction:
    """Predikce, jejíž příznak ``predicted`` plyne ze znaménka skóre.

    Returns:
        Predikce pacienta ``subject_id``, kladná při ``score > 0``.
    """
    return Prediction(
        row_id,
        fold,
        note_id,
        subject_id,
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
