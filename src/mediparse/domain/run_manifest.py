"""Run manifest: z čeho a čím běh vznikl."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.digest import Sha256
from mediparse.domain.environment import Environment
from mediparse.domain.identity import RowId
from mediparse.domain.seed import Seed


class DiagnosticStrip(BaseModel):
    """Záznam ablace D2: které sekce se ze zpráv odstranily a podle jakého lexikonu.

    ``sections`` jsou seřazené klíče sekcí, aby byl JSON deterministický.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sections: Annotated[tuple[str, ...], Field(min_length=1)]
    lexicon_sha256: Sha256


class RunManifest(BaseModel):
    """Provenance tréninku; funkce vstupů a prostředí, bez času běhu.

    ``config_sha256`` je hash configu tréninku, ``seed`` seed modelu. Split
    nese ``split_seed``, ``split_folds`` a ``split_sha256``, otisk přiřazení pacientů do foldů.
    ``diagnostic_strip`` je ``None``, když se zprávy nestripovaly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    dataset_sha256: Sha256
    config_sha256: Sha256
    seed: Seed
    split_seed: Seed
    split_folds: Annotated[int, Field(ge=2)]
    split_sha256: Sha256
    diagnostic_strip: DiagnosticStrip | None
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    environment: Environment
