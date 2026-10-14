"""Kohorta zpráv se štítky pětice: přiřazení diagnóz kódům, trychtýř a manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from mediparse.domain.icd import IcdCode, IcdVersion

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.labels import Diagnosis

type RawCode = tuple[str | None, str | None]


@dataclass(frozen=True)
class CodeLabels:
    """Diagnózy odlišných kódů tak, jak leží v tabulce, a počet neplatných dvojic."""

    diagnoses: Mapping[RawCode, Diagnosis]
    invalid: int


def _parsed(raw: RawCode) -> IcdCode | None:
    version, code = raw
    if version is None or code is None:
        return None
    try:
        return IcdCode(version=IcdVersion(int(version)), code=code)
    except (ValueError, ValidationError):
        return None


def label_codes(raw: Iterable[RawCode], codes: DiagnosisCodes) -> CodeLabels:
    """Přiřadí diagnózu každé odlišné dvojici verze a kódu.

    Dvojice, kterou nelze převést na kód ICD, nemá diagnózu a počítá se jako
    neplatná. Platný kód mimo pětici je v mapování nepřítomný.

    Returns:
        Mapování surových dvojic na diagnózu a počet neplatných dvojic.
    """
    diagnoses: dict[RawCode, Diagnosis] = {}
    invalid = 0
    for pair in set(raw):
        code = _parsed(pair)
        if code is None:
            invalid += 1
            continue
        diagnosis = codes.diagnosis_of(code)
        if diagnosis is not None:
            diagnoses[pair] = diagnosis
    return CodeLabels(diagnoses=diagnoses, invalid=invalid)
