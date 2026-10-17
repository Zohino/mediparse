"""Diagnóza, jejíž přítomnost nese jeden binární label, a jediná akutní diagnóza.

``PerDiagnosis`` je slovník s hodnotou každé diagnózy pětice.
"""

from enum import StrEnum
from typing import Annotated, Final

from pydantic import AfterValidator


class Diagnosis(StrEnum):
    """Diagnóza, jejíž přítomnost nese jeden binární label."""

    DIABETES = "diabetes"
    COPD = "copd"
    CKD = "ckd"
    PNEUMONIA = "pneumonia"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"

    @property
    def chronic(self) -> bool:
        """Chronická diagnóza platí pro všechny zprávy pacienta, akutní jen pro jednu."""
        return self is not ACUTE_DIAGNOSIS


ACUTE_DIAGNOSIS: Final = Diagnosis.PNEUMONIA


def _covers_all_diagnoses[T](value: dict[Diagnosis, T]) -> dict[Diagnosis, T]:
    missing = sorted(diagnosis.value for diagnosis in set(Diagnosis) - set(value))
    if missing:
        msg = f"Chybí diagnózy: {', '.join(missing)}."
        raise ValueError(msg)
    return value


type PerDiagnosis[T] = Annotated[
    dict[Diagnosis, T], AfterValidator(_covers_all_diagnoses)
]
