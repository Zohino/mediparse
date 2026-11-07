"""Prostředí běhu: revize kódu, verze knihoven a hardware stroje."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.digest import CommitSha


class GpuDevice(BaseModel):
    """Model a celková paměť viditelného zařízení GPU."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: Annotated[str, Field(min_length=1)]
    memory_total_bytes: Annotated[int, Field(ge=1)]


class Hardware(BaseModel):
    """CPU, dostupná jádra, RAM hostitele a viditelná GPU.

    Paměť úlohy nese ``max_rss`` benchmarku, nikoli ``memory_bytes``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cpu_model: Annotated[str, Field(min_length=1)]
    logical_cores: Annotated[int, Field(ge=1)]
    available_cores: Annotated[int, Field(ge=1)]
    memory_bytes: Annotated[int, Field(ge=1)]
    gpus: tuple[GpuDevice, ...]


class SourceRevision(BaseModel):
    """Revize zdrojového kódu: commit a příznak rozpracovaného stromu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    commit: CommitSha
    dirty: bool


class Environment(BaseModel):
    """Prostředí běhu: verze knihoven a revize kódu, společné všem manifestům."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    packages: dict[str, str]
    source: SourceRevision
