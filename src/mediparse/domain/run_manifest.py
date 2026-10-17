"""Run manifest: z čeho a čím běh vznikl."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mediparse.domain.digest import Sha256
from mediparse.domain.environment import Environment
from mediparse.domain.identity import RowId


class RunManifest(BaseModel):
    """Provenance tréninku; funkce vstupů a prostředí, bez času běhu.

    ``config_sha256`` je hash configu tréninku, ``seed`` seed modelu. Split
    nese ``split_seed`` a ``split_sha256``, otisk přiřazení pacientů do foldů.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    dataset_sha256: Sha256
    config_sha256: Sha256
    seed: int
    split_seed: int
    split_sha256: Sha256
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    environment: Environment
