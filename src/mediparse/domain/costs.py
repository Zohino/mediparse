"""Agregované náklady inference a percentil metodou nejbližšího pořadí."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mediparse.domain.environment import GpuDevice, Hardware

if TYPE_CHECKING:
    from collections.abc import Sequence

PERCENT_SCALE: Final = 100


class LatencyPlan(BaseModel):
    """Počet zahřívacích volání a nejvýše měřených zpráv."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    warmup: Annotated[int, Field(ge=0)]
    samples: Annotated[int, Field(ge=1)]


class Latency(BaseModel):
    """Latence jednotlivých zpráv v sekundách po zahřátí modelu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    samples: Annotated[int, Field(ge=1)]
    p50_seconds: Annotated[float, Field(ge=0)]
    p95_seconds: Annotated[float, Field(ge=0)]


class BatchThroughput(BaseModel):
    """Počet zpráv dávky a její čas; podíl dopočítá konzument."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    notes: Annotated[int, Field(ge=1)]
    seconds: Annotated[float, Field(ge=0)]


class GpuUsage(BaseModel):
    """Energie a paměť celého zařízení za okno měření, včetně výchozího stavu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    device: GpuDevice
    energy_joules: Annotated[float, Field(ge=0)] | None
    memory_used_start_bytes: Annotated[int, Field(ge=0)]
    memory_used_peak_bytes: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _peak_covers_start(self) -> Self:
        if self.memory_used_peak_bytes < self.memory_used_start_bytes:
            msg = "Paměťová špička nesmí být menší než na začátku měření."
            raise ValueError(msg)
        return self


class GpuMeasurement(BaseModel):
    """Délka okna a měření zařízení v pořadí viditelnosti."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    seconds: Annotated[float, Field(ge=0)]
    devices: tuple[GpuUsage, ...]


class TrainingCosts(BaseModel):
    """Měření GPU za celý use case tréninku, mimo deterministický manifest."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    gpu: GpuMeasurement


class InferenceCosts(BaseModel):
    """Plán a agregovaná měření se strojem inference, bez textu a identifikátorů."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    plan: LatencyPlan
    latency: Latency
    batch: BatchThroughput
    hardware: Hardware
    gpu: GpuMeasurement

    @model_validator(mode="after")
    def _measurement_matches_hardware(self) -> Self:
        if tuple(usage.device for usage in self.gpu.devices) != self.hardware.gpus:
            msg = "Měřená zařízení musí odpovídat GPU hardwaru ve stejném pořadí."
            raise ValueError(msg)
        return self


def nearest_rank(values: Sequence[float], percent: float) -> float:
    """Vybere ceil(percent * n / 100)-tou nejmenší hodnotu bez interpolace.

    Returns:
        Skutečná vstupní hodnota na příslušném pořadí.

    Raises:
        ValueError: Vstup je prázdný nebo percent není v (0, 100].
    """
    if not values or not 0 < percent <= PERCENT_SCALE:
        msg = "Percentil vyžaduje neprázdný vstup a percent v (0, 100]."
        raise ValueError(msg)
    return sorted(values)[ceil(percent * len(values) / PERCENT_SCALE) - 1]


def latency(durations: Sequence[float]) -> Latency:
    """Shrne měřené doby do počtu vzorků a percentilů p50 a p95.

    Returns:
        Agregát latencí v sekundách.
    """
    return Latency(
        samples=len(durations),
        p50_seconds=nearest_rank(durations, 50),
        p95_seconds=nearest_rank(durations, 95),
    )
