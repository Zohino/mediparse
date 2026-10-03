"""Provenance syntetického korpusu: čím a podle čeho korpus vznikl a který audit ho pustil do repozitáře."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from mediparse.domain.corpus_audit import CommitSha, Sha256

ModelId = Annotated[str, Field(pattern=r"^claude-[a-z0-9]+(-[a-z0-9]+)*$")]
Version = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]


class Generation(BaseModel):
    """Vstupy generování: seed a otisk configu plánů, šablona, model, nástroj, den a verze specifikace."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: NonNegativeInt
    sampler_config_sha256: Sha256
    verbalization_template_sha256: Sha256
    model: ModelId
    claude_code_version: Version
    generated_on: date
    specification_commit: CommitSha


class ProvenanceRecord(BaseModel):
    """Záznam provenance korpusu: generování a otisk záznamu čistého auditu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    generation: Generation
    audit_sha256: Sha256


class InvalidProvenanceRecordError(ValueError):
    """Záznam provenance neodpovídá schématu."""


def provenance_violations(
    record: ProvenanceRecord | None, audit_sha256: str
) -> tuple[str, ...]:
    """Důvody, proč provenance auditovaného korpusu neplatí; prázdný výsledek znamená, že platí.

    Returns:
        Popis chybějícího záznamu nebo záznamu, který ukazuje na jiný audit.
    """
    if record is None:
        return ("Korpus nemá záznam provenance.",)
    if record.audit_sha256 != audit_sha256:
        return (
            "Provenance ukazuje na jiný audit, po novém auditu je nutné ji zapsat znovu.",
        )
    return ()
