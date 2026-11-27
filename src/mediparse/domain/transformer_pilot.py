"""Podmínky pilotu transformeru a agregáty jeho epoch."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.representation import Aggregation, EncoderSpec
from mediparse.domain.seed import Seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.split import SplitConfig
    from mediparse.domain.tuning import ValidationScores


class PilotTraining(BaseModel):
    """Společná optimalizace všech běhů pilotu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    epochs: PositiveInt
    batch_tokens: PositiveInt
    lr: PositiveFloat
    weight_decay: NonNegativeFloat
    warmup_ratio: Annotated[float, Field(ge=0, lt=1)]


class PilotRun(BaseModel):
    """Jedna varianta encoderu, spojení oken a podílu pacientů."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    encoder: EncoderSpec
    aggregation: Aggregation
    freeze_embeddings: bool
    fraction: Annotated[float, Field(gt=0, le=1)]


class TransformerPilotConfig(BaseModel):
    """Striktní podmínky pilotu na jediném vývojovém foldu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    seed: Seed
    subsample_parts: PositiveInt
    folds: Annotated[int, Field(ge=2)]
    stopping_parts: Annotated[int, Field(ge=2)]
    strip_diagnostic_sections: bool
    training: PilotTraining
    runs: Annotated[tuple[PilotRun, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _valid_runs(self) -> Self:
        if len({run.encoder.name for run in self.runs}) != len(self.runs):
            msg = "Běhy pilotu musí mít jedinečná jména encoderů."
            raise ValueError(msg)
        if any(
            run.encoder.text_preprocessing != TextPreprocessing.NONE
            for run in self.runs
        ):
            msg = "Pilot vyžaduje text_preprocessing none."
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class PilotEpoch:
    """Měření epochy a validační skóre v pořadí vstupu."""

    epoch: int
    seconds: float
    validation_seconds: float
    peak_memory_bytes: int
    stop_loss: float
    scores: ValidationScores


@dataclass(frozen=True)
class PilotRecord:
    """Veřejný agregát jedné epochy bez řádkových dat."""

    run: str
    model: str
    window: int
    aggregation: Aggregation
    freeze_embeddings: bool
    fraction: float
    epoch: int
    fit_notes: int
    fit_patients: int
    stop_notes: int
    validation_notes: int
    seconds: float
    validation_seconds: float
    peak_memory_bytes: int
    stop_loss: float
    macro_f1: float


@dataclass(frozen=True)
class TransformerPilotReport:
    """Dokončené agregáty a původ vývojového vzorku."""

    records: tuple[PilotRecord, ...]
    sample_notes: int
    sample_patients: int
    development_notes: int
    development_patients: int
    split_config: SplitConfig
    split_sha256: str
    seconds: float


def best_pilot_epoch(records: Sequence[PilotRecord]) -> PilotRecord | None:
    """Vybere epochu se stejnou podmínkou konečné stop BCE jako klasifikátor.

    Returns:
        Epocha s nejnižší konečnou ztrátou, nebo None při úplné divergenci.
    """
    finite = [record for record in records if isfinite(record.stop_loss)]
    return min(finite, key=lambda record: record.stop_loss) if finite else None
