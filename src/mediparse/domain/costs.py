"""Agregované náklady inference a percentil metodou nejbližšího pořadí."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.environment import Hardware

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


class InferenceCosts(BaseModel):
    """Plán a agregovaná měření se strojem inference, bez textu a identifikátorů."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    plan: LatencyPlan
    latency: Latency
    batch: BatchThroughput
    hardware: Hardware


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
