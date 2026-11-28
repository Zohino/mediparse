"""Diagnóza, jejíž přítomnost nese jeden binární label.

``PerDiagnosis`` je typ pro hodnotu každé diagnózy pětice; pydantic ho načte jako ``dict``.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator


class Diagnosis(StrEnum):
    """Diagnóza, jejíž přítomnost nese jeden binární label."""

    DIABETES = "diabetes"
    COPD = "copd"
    CKD = "ckd"
    PNEUMONIA = "pneumonia"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"


def _covers_all_diagnoses[T](value: Mapping[Diagnosis, T]) -> Mapping[Diagnosis, T]:
    missing = sorted(diagnosis.value for diagnosis in set(Diagnosis) - set(value))
    if missing:
        msg = f"Chybí diagnózy: {', '.join(missing)}."
        raise ValueError(msg)
    return value


type PerDiagnosis[T] = Annotated[
    Mapping[Diagnosis, T], AfterValidator(_covers_all_diagnoses)
]
