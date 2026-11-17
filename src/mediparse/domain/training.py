"""Trénink modelu: config a výsledek běhu pro jednu diagnózu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.identity import RowId
from mediparse.domain.model_input import ModelInput
from mediparse.domain.seed import Seed
from mediparse.domain.tuning import ParamValue

if TYPE_CHECKING:
    from mediparse.domain.split import SplitConfig


class TrainingConfig(BaseModel):
    """Parametry tréninku: řádek matice, diagnóza, seed modelu, testovací fold a vnitřní CV.

    ``input`` určuje ablaci diagnostických sekcí a úroveň předzpracování textu TF-IDF.
    ``fold`` je číslo foldu, který slouží jako test; jeho horní mez zná až ``Split``.
    ``parameters`` jsou pevné jen pro běh bez ladění, jinak se čtou z manifestu.
    ``threshold_folds`` řídí vnitřní CV prahu po pacientech.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    diagnosis: Diagnosis
    seed: Seed
    fold: Annotated[int, Field(ge=0)]
    parameters: dict[str, ParamValue] | None
    threshold_folds: Annotated[int, Field(ge=2)]
    input: ModelInput


@dataclass(frozen=True)
class TrainingReport:
    """Výsledek tréninku: velikosti částí, původ splitu, práh a konvergence."""

    diagnosis: Diagnosis
    train_notes: int
    test_notes: int
    test_positives: int
    split_config: SplitConfig
    split_sha256: str
    threshold: float
    converged: bool
