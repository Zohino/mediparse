"""Provenance syntetického korpusu: čím a podle čeho korpus vznikl a který audit ho pustil do repozitáře."""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, NonNegativeInt

from mediparse.domain.corpus_audit import CommitSha, Sha256, fingerprint
from mediparse.domain.verbalization import render_prompt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig

_MODEL_ID: Final = re.compile(r"claude-[a-z0-9]+(-[a-z0-9]+)*")
_VERSION: Final = re.compile(r"\d+\.\d+\.\d+")


def model_id(value: str) -> str:
    """Ověří, že hodnota je identifikátor modelu Claude.

    Returns:
        Hodnota beze změny.

    Raises:
        ValueError: Hodnota není identifikátor modelu Claude.
    """
    if _MODEL_ID.fullmatch(value) is None:
        msg = "Model musí být identifikátor modelu Claude, například claude-opus-5-5."
        raise ValueError(msg)
    return value


def version(value: str) -> str:
    """Ověří, že hodnota je verze ve tvaru major.minor.patch.

    Returns:
        Hodnota beze změny.

    Raises:
        ValueError: Hodnota nemá tvar major.minor.patch.
    """
    if _VERSION.fullmatch(value) is None:
        msg = "Verze musí mít tvar major.minor.patch, například 2.1.5."
        raise ValueError(msg)
    return value


ModelId = Annotated[str, AfterValidator(model_id)]
Version = Annotated[str, AfterValidator(version)]


class Generation(BaseModel):
    """Vstupy generování: seed a otisk configu plánů, otisk zadání, model, nástroj, den a verze specifikace."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: NonNegativeInt
    sampler_config_sha256: Sha256
    prompts_sha256: Sha256
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


def prompts_sha256(
    template: str, plans: Iterable[NotePlan], config: SamplerConfig
) -> str:
    """Otisk zadání všech zpráv: dvojice note_id a text, který model dostane.

    Returns:
        SHA-256 otisk; mění se s každou změnou zadání, i s větou, kterou skládá kód.
    """
    return fingerprint(
        (plan.note_id, render_prompt(template, plan, config).encode()) for plan in plans
    )


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
