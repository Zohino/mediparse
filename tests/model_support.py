"""Sdílená data testů modelů: predikce a config tréninku ve tvaru doménových typů."""

from __future__ import annotations

import json
from typing import Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Prediction
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.domain.note import note_id
from mediparse.domain.smoketest_training import TrainingConfig
from mediparse.domain.split import SplitConfig

SEPARABLE_VOCABULARY: Final = (
    "diabetes insulin glucose metformin hyperglycemia",
    "fracture cast orthopedic surgery splint",
)

OVERLAPPING: Final = (
    "diabetes insulin glucose clinic review",
    "fracture cast splint clinic review",
)


def separable_notes(
    patients: int, vocabulary: tuple[str, str] = SEPARABLE_VOCABULARY
) -> list[InputNote]:
    """Zprávy pacientů střídavě s diabetem a bez něj, pacient má 1 až 7 zpráv nepravidelně.

    Args:
        patients: Počet pacientů s ``subject_id`` od 1; diabetes má každý druhý
            počínaje prvním.
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
                note_id(patient + 1, visit),
                patient + 1,
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
        Slovník s klíči ``row_id``, ``diagnosis``, ``seed``, ``fold`` a
        ``regularization``, přepsanými hodnotami z ``overrides``.
    """
    return {
        "row_id": "row",
        "diagnosis": "diabetes",
        "seed": 0,
        "fold": 0,
        "regularization": 1.0,
    } | overrides


def parsed_training_config(**overrides: object) -> TrainingConfig:
    """Config tréninku z ``training_config`` rozparsovaný jako v produkci.

    Returns:
        ``TrainingConfig`` s přepsanými hodnotami z ``overrides``.
    """
    return TrainingConfig.model_validate_json(json.dumps(training_config(**overrides)))


def split_config_json(**overrides: object) -> dict[str, object]:
    """Platný config splitu ve tvaru JSON; test změní jen pole, o kterém je.

    Returns:
        Slovník s klíči ``seed`` a ``folds``, přepsanými hodnotami z ``overrides``.
    """
    return {"seed": 0, "folds": 2} | overrides


def split_config(**overrides: object) -> SplitConfig:
    """Config splitu z ``split_config_json`` rozparsovaný jako v produkci.

    Returns:
        ``SplitConfig`` s přepsanými hodnotami z ``overrides``.
    """
    return SplitConfig.model_validate_json(json.dumps(split_config_json(**overrides)))
