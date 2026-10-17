"""Kód ICD obou verzí a přiřazení kódu k diagnóze pětice podle předpon."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import IntEnum
from itertools import combinations
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from mediparse.domain.diagnosis import Diagnosis, PerDiagnosis

_CODE: Final = re.compile(r"^[0-9A-Z]+$")


class IcdVersion(IntEnum):
    """Verze klasifikace, ve které je kód zapsaný."""

    NINE = 9
    TEN = 10


def _normalized(code: str) -> str:
    normalized = code.strip().upper()
    if not _CODE.fullmatch(normalized):
        msg = f"Kód ICD smí obsahovat jen číslice a písmena bez teček: {code!r}."
        raise ValueError(msg)
    return normalized


class IcdCode(BaseModel):
    """Kód ICD bez teček, jak ho zapisuje MIMIC i GEMs."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: IcdVersion
    code: str

    @field_validator("code")
    @classmethod
    def _valid_code(cls, code: str) -> str:
        return _normalized(code)


class DiagnosisCodes(BaseModel):
    """Předpony kódů každé diagnózy podle verze; kód nepatří do dvou diagnóz."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    prefixes: PerDiagnosis[Mapping[IcdVersion, tuple[str, ...]]]

    @model_validator(mode="after")
    def _valid_prefixes(self) -> Self:
        self._check_nonempty()
        self._check_normalized()
        self._check_disjoint()
        return self

    def _check_nonempty(self) -> None:
        for diagnosis, by_version in self.prefixes.items():
            if not any(by_version.values()):
                msg = f"Diagnóza {diagnosis.value} nemá žádnou předponu."
                raise ValueError(msg)

    def _check_normalized(self) -> None:
        for by_version in self.prefixes.values():
            for prefix in (p for ps in by_version.values() for p in ps):
                if _normalized(prefix) != prefix:
                    msg = f"Předpona {prefix!r} není normalizovaná (velká písmena, bez mezer)."
                    raise ValueError(msg)

    def _check_disjoint(self) -> None:
        for version in IcdVersion:
            owned = [
                (diagnosis, prefix)
                for diagnosis, by_version in self.prefixes.items()
                for prefix in by_version.get(version, ())
            ]
            for (first, a), (second, b) in combinations(owned, 2):
                if first is second or not (a.startswith(b) or b.startswith(a)):
                    continue
                msg = (
                    f"Předpony {a} ({first.value}) a {b} ({second.value}) "
                    f"se v ICD-{version.value} překrývají."
                )
                raise ValueError(msg)

    def diagnosis_of(self, code: IcdCode) -> Diagnosis | None:
        """Najde diagnózu, jejíž předpona v téže verzi začíná kód.

        Returns:
            Diagnóza nebo None, když kód nepatří do žádné skupiny.
        """
        for diagnosis, by_version in self.prefixes.items():
            if code.code.startswith(by_version.get(code.version, ())):
                return diagnosis
        return None
