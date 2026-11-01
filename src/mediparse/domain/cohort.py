"""Kohorta zpráv se štítky pětice: přiřazení diagnóz kódům, trychtýř a manifest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Self

from pydantic import BaseModel, ConfigDict, model_validator

from mediparse.domain.diagnosis import Diagnosis, PerDiagnosis
from mediparse.domain.digest import Sha256
from mediparse.domain.environment import Environment
from mediparse.domain.icd import DiagnosisCodes, IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Dostatek pozitivních i negativních zpráv zajišťuje stabilitu splitu a metrik.
MIN_CELL: Final = 10

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
    except ValueError:
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


class FunnelStepName(StrEnum):
    """Kroky výběru zpráv v pořadí trychtýře sešitu 5."""

    ALL_NOTES = "all_notes"
    WITH_ICD = "with_icd"
    LONG_ENOUGH = "long_enough"
    UNIQUE = "unique"


class FunnelStep(BaseModel):
    """Počet zpráv a pacientů, kteří krokem trychtýře prošli."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    step: FunnelStepName
    notes: int
    patients: int


class CohortCounts(BaseModel):
    """Agregáty kohorty: trychtýř ve čtyřech krocích a pozitivní zprávy každé diagnózy."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    funnel: tuple[FunnelStep, ...]
    positives: PerDiagnosis[int]

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if tuple(step.step for step in self.funnel) != tuple(FunnelStepName):
            msg = "Kroky trychtýře musí být čtyři v pořadí sešitu 5."
            raise ValueError(msg)
        for before, after in zip(self.funnel, self.funnel[1:], strict=False):
            if after.notes > before.notes or after.patients > before.patients:
                msg = f"Kroky trychtýře nesmí růst: {after.step.value}."
                raise ValueError(msg)
        return self


def ensure_publishable(counts: CohortCounts) -> None:
    """Ověří, že kohorta není prázdná a každá diagnóza má dost obou tříd.

    Args:
        counts: Agregáty kohorty.

    Raises:
        InvalidInputError: Kohorta je prázdná nebo má diagnóza pod ``MIN_CELL``
            pozitivních nebo negativních zpráv; hláška jmenuje diagnózu bez počtu.
    """
    notes = counts.funnel[-1].notes
    if notes == 0:
        msg = "Kohorta je prázdná."
        raise InvalidInputError(msg)
    for diagnosis in Diagnosis:
        positive = counts.positives[diagnosis]
        if min(positive, notes - positive) < MIN_CELL:
            msg = (
                f"Diagnóza {diagnosis.value} nemá v kohortě aspoň {MIN_CELL} "
                "pozitivních a negativních zpráv."
            )
            raise InvalidInputError(msg)


class CohortManifest(BaseModel):
    """Provenance kohorty: otisky vstupů a výstupu, prostředí, revize a agregáty."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cohort_sha256: Sha256
    config_sha256: Sha256
    sources: dict[str, Sha256]
    environment: Environment
    counts: CohortCounts
    invalid_codes: int
