"""Diagnóza, jejíž přítomnost nese jeden binární label, a jediná akutní diagnóza."""

from enum import StrEnum
from typing import Final


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
