"""Run manifest: z čeho a čím běh vznikl."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.digest import Sha256
from mediparse.domain.environment import Environment, Hardware
from mediparse.domain.identity import RowId
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.seed import Seed


class DiagnosticStrip(BaseModel):
    """Záznam ablace D2: které sekce se ze zpráv odstranily a podle jakého lexikonu.

    ``sections`` jsou seřazené klíče sekcí bez opakování, aby byl JSON deterministický.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sections: Annotated[tuple[str, ...], Field(min_length=1)]
    lexicon_sha256: Sha256

    @field_validator("sections", mode="before")
    @classmethod
    def _sorted_unique(cls, sections: object) -> object:
        if not isinstance(sections, (tuple, list)):
            return sections
        return tuple(sorted(set(sections)))


class RunManifest(BaseModel):
    """Provenance tréninku; funkce vstupů a prostředí, bez času běhu.

    ``config_sha256`` je hash configu tréninku, ``seed`` seed modelu. Split
    nese ``split_seed``, ``split_folds`` a ``split_sha256``, otisk přiřazení pacientů do foldů.
    ``diagnostic_strip`` je ``None``, když se zprávy nestripovaly.
    ``normalize_clinical_text`` je ``None``, když vstup TF-IDF neprošel normalizací,
    jinak slovníky zkratek, variant a překlepů, které dostal trenér.
    Prázdné ``variants`` znamená vypnutý přepínač přepisu variant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    dataset_sha256: Sha256
    predictions_sha256: Sha256
    config_sha256: Sha256
    seed: Seed
    split_seed: Seed
    split_folds: Annotated[int, Field(ge=2)]
    split_sha256: Sha256
    diagnostic_strip: DiagnosticStrip | None
    normalize_clinical_text: TextNormalization | None
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    environment: Environment
    hardware: Hardware


class EvaluationManifest(BaseModel):
    """Provenance evaluace: ověřené vstupy, otisk metrik a prostředí evaluátoru."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    training_manifest_sha256: Sha256
    predictions_sha256: Sha256
    dataset_sha256: Sha256
    split_sha256: Sha256
    metrics_sha256: Sha256
    environment: Environment


def ensure_training_lineage(
    training: RunManifest,
    *,
    predictions_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
) -> None:
    """Ověří, že vstupy evaluace pocházejí z dodaného tréninku.

    Raises:
        InvalidInputError: Otisky nesedí; zpráva uvádí všechna pole s rozdílem.
    """
    actual = {
        "predictions_sha256": predictions_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
    }
    mismatched = [
        field for field, digest in actual.items() if getattr(training, field) != digest
    ]
    if mismatched:
        msg = f"Vstupy evaluace neodpovídají tréninku: {', '.join(mismatched)}."
        raise InvalidInputError(msg)
