"""Evaluace: predikce zpráv, jejich skóre a metriky na diagnózu a řádek matice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.labels import Diagnosis


@dataclass(frozen=True)
class Classification:
    """Výstup klasifikátoru pro jeden text: label a rozhodovací skóre."""

    predicted: bool
    score: float


@dataclass(frozen=True)
class Prediction:
    """Predikce jedné diagnózy pro jednu testovací zprávu řádku matice."""

    row_id: str
    fold: int
    note_id: str
    subject_id: int
    diagnosis: Diagnosis
    truth: bool
    predicted: bool
    score: float


@dataclass(frozen=True)
class BinaryMetrics:
    """Held-out metriky binárního klasifikátoru pro pozitivní třídu."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class DiagnosisMetrics:
    """Metriky jedné diagnózy na jednom řádku matice s velikostí testu."""

    row_id: str
    diagnosis: Diagnosis
    test_notes: int
    test_positives: int
    metrics: BinaryMetrics


def ensure_both_classes(diagnosis: Diagnosis, truth: Sequence[bool]) -> None:
    """Ověří, že pravda diagnózy obsahuje pozitivní i negativní zprávu.

    Args:
        diagnosis: Diagnóza, pro kterou se počítají metriky.
        truth: Pravda testovacích zpráv.

    Raises:
        InvalidInputError: Pravda má jedinou třídu, AUC by nebyla definovaná.
    """
    positives = sum(truth)
    if 0 < positives < len(truth):
        return
    msg = (
        f"Diagnóza {diagnosis}: {positives} pozitivních z {len(truth)} zpráv, "
        "metriky potřebují obě třídy."
    )
    raise InvalidInputError(msg)


def ensure_unique_keys(predictions: Sequence[Prediction]) -> None:
    """Ověří, že klíč (row_id, fold, note_id, diagnóza) se v predikcích neopakuje.

    Args:
        predictions: Predikce ke zpracování.

    Raises:
        InvalidInputError: Některý klíč se opakuje.
    """
    seen: set[tuple[str, int, str, Diagnosis]] = set()
    for item in predictions:
        key = (item.row_id, item.fold, item.note_id, item.diagnosis)
        if key in seen:
            msg = (
                f"Opakovaná predikce: řádek {item.row_id}, fold {item.fold}, "
                f"zpráva {item.note_id}, diagnóza {item.diagnosis}."
            )
            raise InvalidInputError(msg)
        seen.add(key)
