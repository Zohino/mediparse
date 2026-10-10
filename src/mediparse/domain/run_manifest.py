"""Run manifest: z čeho a čím běh vznikl."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mediparse.domain.corpus_audit import CommitSha, Sha256


class SourceRevision(BaseModel):
    """Revize zdrojového kódu: commit a příznak rozpracovaného stromu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    commit: CommitSha
    dirty: bool


class RunManifest(BaseModel):
    """Provenance tréninku; funkce vstupů, prostředí a revize, bez času běhu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: str
    dataset_sha256: Sha256
    config_sha256: Sha256
    seed: int
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    packages: dict[str, str]
    source: SourceRevision
