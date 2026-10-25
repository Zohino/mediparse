"""Načtení předpon kódů ICD pětice z configu kandidátů, který sdílí EDA a kohorta."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


class _Prefixes(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    icd9_prefixes: tuple[str, ...]
    icd10_prefixes: tuple[str, ...]


class Candidates(BaseModel):
    """Kandidáti configu podle klíče diagnózy; pole kandidátů čtou až adaptéry."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    candidates: dict[str, dict[str, Any]]


def load_diagnosis_codes(path: Path) -> DiagnosisCodes:
    """Přečte předpony diagnóz pětice; ostatní kandidáty a pole ignoruje.

    Returns:
        Předpony všech diagnóz, které prošly doménovými invarianty.
    """
    return parse_file(path, _parse)


def _parse(text: str) -> DiagnosisCodes:
    candidates = Candidates.model_validate_json(text).candidates
    parsed = {
        diagnosis: _Prefixes.model_validate(candidates[diagnosis.value])
        for diagnosis in Diagnosis
        if diagnosis.value in candidates
    }
    return DiagnosisCodes(
        prefixes={
            diagnosis: {
                IcdVersion.NINE: prefixes.icd9_prefixes,
                IcdVersion.TEN: prefixes.icd10_prefixes,
            }
            for diagnosis, prefixes in parsed.items()
        }
    )
