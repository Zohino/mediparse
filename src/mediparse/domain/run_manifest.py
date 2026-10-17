"""Run manifest: z čeho a čím běh vznikl."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mediparse.domain.digest import Sha256
from mediparse.domain.environment import Environment
from mediparse.domain.identity import RowId


class RunManifest(BaseModel):
    """Provenance tréninku; funkce vstupů a prostředí, bez času běhu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    dataset_sha256: Sha256
    config_sha256: Sha256
    seed: int
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    environment: Environment
