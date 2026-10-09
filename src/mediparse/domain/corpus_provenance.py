"""Provenance syntetického korpusu: čím a podle čeho korpus vznikl a který audit ho pustil do repozitáře."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
)

from mediparse.domain.corpus_audit import CommitSha, Sha256, fingerprint
from mediparse.domain.verbalization import render_prompt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig

type Masking = Literal["[[n]]"]

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
        msg = "Model musí být identifikátor modelu Claude, například claude-sonnet-5-5."
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


class Decoding(BaseModel):
    """Dekódování překladu: teplota a tokeny, které generování ukončují."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    temperature: float
    stop_token_ids: tuple[int, ...]


class TranslationRun(BaseModel):
    """Běh překladu: začátek, otisky požadavků a skriptu, grafická karta, CUDA a verze balíčků."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    started: datetime
    requests_sha256: Sha256
    script_sha256: Sha256
    gpu: str
    cuda: str
    packages: dict[str, str]


class Translation(BaseModel):
    """Vstupy překladu: model, revize, dekódování, otisk požadavků, použité běhy, neshody značek ___ a maskování."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str
    revision: CommitSha
    dtype: str
    window: PositiveInt
    decoding: Decoding
    requests_sha256: Sha256
    runs: tuple[TranslationRun, ...]
    marker_mismatches: tuple[str, ...]
    masking: Masking | None = None


class ProvenanceRecord(BaseModel):
    """Záznam provenance korpusu: generování, volitelně překlad a otisk záznamu čistého auditu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    generation: Generation
    audit_sha256: Sha256
    translation: Translation | None = None


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
    record: ProvenanceRecord | None, audit_sha256: str, *, translated: bool
) -> tuple[str, ...]:
    """Důvody, proč provenance auditovaného korpusu neplatí; prázdný výsledek znamená, že platí.

    Args:
        record: Záznam provenance, nebo None, když chybí.
        audit_sha256: Otisk souboru se záznamem auditu.
        translated: Zda korpus obsahuje české zprávy.

    Returns:
        Popis chybějícího záznamu, záznamu, který ukazuje na jiný audit, nebo českého
        korpusu bez provenance překladu.
    """
    if record is None:
        return ("Korpus nemá záznam provenance.",)
    if record.audit_sha256 != audit_sha256:
        return (
            "Provenance ukazuje na jiný audit, po novém auditu je nutné ji zapsat znovu.",
        )
    if translated and record.translation is None:
        return ("Český korpus nemá provenance překladu.",)
    return ()
