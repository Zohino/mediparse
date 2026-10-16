"""Načtení předpon kódů ICD pětice z configu kandidátů, který sdílí EDA a kohorta."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.input_file import parse_file

DIAGNOSIS_CANDIDATES_PATH: Final = Path("config/diagnosis_candidates.json")


class _Prefixes(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    icd9_prefixes: tuple[str, ...]
    icd10_prefixes: tuple[str, ...]


class _Candidates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    candidates: dict[str, dict[str, Any]]


def load_diagnosis_codes(path: Path) -> DiagnosisCodes:
    """Přečte předpony diagnóz pětice; ostatní kandidáty a pole ignoruje.

    Returns:
        Předpony všech diagnóz, které prošly doménovými invarianty.
    """
    return parse_file(path, _parse)


def _parse(text: str) -> DiagnosisCodes:
    candidates = _Candidates.model_validate_json(text).candidates
    missing = [d.value for d in Diagnosis if d.value not in candidates]
    if missing:
        msg = f"V configu kandidátů chybí diagnózy: {', '.join(missing)}."
        raise InvalidInputError(msg)
    parsed = {
        diagnosis: _Prefixes.model_validate(candidates[diagnosis.value])
        for diagnosis in Diagnosis
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
