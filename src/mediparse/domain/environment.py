"""Prostředí běhu: revize kódu a verze knihoven."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mediparse.domain.digest import CommitSha


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
