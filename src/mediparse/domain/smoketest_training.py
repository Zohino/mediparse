"""Trénink smoketestu: config a výsledek běhu pro jednu diagnózu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.identity import RowId


class TrainingConfig(BaseModel):
    """Parametry tréninku: řádek matice, diagnóza, seed odložení, foldy a regularizace.

    ``fold`` je číslo foldu, který slouží jako test; jeho horní mez zná až ``Split``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    diagnosis: Diagnosis
    seed: int
    folds: Annotated[int, Field(ge=2)]
    fold: Annotated[int, Field(ge=0)]
    regularization: Annotated[float, Field(gt=0.0)]


@dataclass(frozen=True)
class TrainingReport:
    """Výsledek běhu: velikosti částí a počet pozitivních v testu."""

    diagnosis: Diagnosis
    train_notes: int
    test_notes: int
    test_positives: int
