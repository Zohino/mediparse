"""Labely pěti diagnóz: prevalence, společný výskyt a jejich společné rozdělení s maximální entropií."""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
type LabelSet = frozenset[Diagnosis]

_SYMMETRY_TOLERANCE: Final = 1e-3


class Diagnosis(StrEnum):
    """Diagnóza, jejíž přítomnost nese jeden binární label."""

    DIABETES = "diabetes"
    CKD = "ckd"
    HEART_FAILURE = "heart_failure"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"
    AKI = "aki"

    @property
    def chronic(self) -> bool:
        """Chronická diagnóza platí pro všechny zprávy pacienta, akutní jen pro jednu."""
        return self is not Diagnosis.AKI


class LabelModel(BaseModel):
    """Prevalence diagnóz a podmíněné pravděpodobnosti jejich společného výskytu z EDA."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    prevalence: dict[Diagnosis, Probability]
    conditional: dict[Diagnosis, dict[Diagnosis, Probability]]
    prevalence_tolerance: Probability
    ipf_tolerance: Annotated[float, Field(gt=0.0)]
    ipf_max_passes: PositiveInt

    @model_validator(mode="after")
    def _covers_all_diagnoses(self) -> Self:
        expected = set(Diagnosis)
        rows = [set(row) for row in self.conditional.values()]
        if set(self.prevalence) != expected or set(self.conditional) != expected:
            msg = "Prevalence i podmíněná matice musí pokrývat všech pět diagnóz."
            raise ValueError(msg)
        if any(row != expected for row in rows):
            msg = "Každý řádek podmíněné matice musí pokrývat všech pět diagnóz."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _conditionals_agree(self) -> Self:
        for first, second in combinations(Diagnosis, 2):
            forward = self.co_occurrence(first, second)
            backward = self.co_occurrence(second, first)
            if abs(forward - backward) > _SYMMETRY_TOLERANCE:
                msg = f"P({first}, {second}) se z obou směrů matice liší: {forward} a {backward}."
                raise ValueError(msg)
        return self

    def co_occurrence(self, first: Diagnosis, second: Diagnosis) -> float:
        """Pravděpodobnost, že zpráva nese obě diagnózy.

        Returns:
            Součin prevalence první diagnózy a podmíněné pravděpodobnosti druhé.
        """
        return self.prevalence[first] * self.conditional[first][second]


def label_shares(label_sets: Sequence[Collection[Diagnosis]]) -> dict[Diagnosis, float]:
    """Podíl zpráv s každým labelem.

    Returns:
        Slovník diagnóza → podíl zpráv, které label nesou.
    """
    return {
        diagnosis: sum(diagnosis in labels for labels in label_sets) / len(label_sets)
        for diagnosis in Diagnosis
    }


def prevalence_outliers(
    shares: Mapping[Diagnosis, float], model: LabelModel
) -> tuple[Diagnosis, ...]:
    """Diagnózy, jejichž podíl leží dál od cílové prevalence než tolerance modelu.

    Returns:
        Diagnózy mimo toleranci v pořadí configu.
    """
    return tuple(
        diagnosis
        for diagnosis, target in model.prevalence.items()
        if abs(shares[diagnosis] - target) > model.prevalence_tolerance
    )


class IpfNotConvergedError(ValueError):
    """Iterativní proporcionální přizpůsobení nedosáhlo tolerance v povoleném počtu průchodů."""


def label_sets() -> tuple[LabelSet, ...]:
    """Všech 32 kombinací pozitivních diagnóz v pevném pořadí.

    Returns:
        Množiny pozitivních diagnóz seřazené podle bitové masky v pořadí výčtu.
    """
    diagnoses = tuple(Diagnosis)
    return tuple(
        frozenset(d for bit, d in enumerate(diagnoses) if mask >> bit & 1)
        for mask in range(2 ** len(diagnoses))
    )


def max_entropy_joint(model: LabelModel) -> dict[LabelSet, float]:
    """Společné rozdělení labelů s maximální entropií při daných prevalencích a párových výskytech.

    Iterativní proporcionální přizpůsobení (Deming a Stephan 1940) začíná
    z rovnoměrného rozdělení; jeho limitou je I-projekce na množinu rozdělení
    s danými marginálami, tedy rozdělení s maximální entropií (Csiszár 1975).

    Returns:
        Pravděpodobnost každé kombinace pozitivních diagnóz.

    Raises:
        IpfNotConvergedError: Rozdělení nesplnilo marginály v povoleném počtu průchodů.
    """
    cells = label_sets()
    joint = dict.fromkeys(cells, 1 / len(cells))
    tables = [
        _pair_table(model, first, second)
        for first, second in combinations(Diagnosis, 2)
    ]
    for _ in range(model.ipf_max_passes):
        for table in tables:
            _fit(joint, table)
        if max(_deviation(joint, table) for table in tables) <= model.ipf_tolerance:
            return joint
    msg = f"IPF nedosáhlo tolerance {model.ipf_tolerance} za {model.ipf_max_passes} průchodů."
    raise IpfNotConvergedError(msg)


def acute_given_chronic(joint: Mapping[LabelSet, float]) -> dict[LabelSet, float]:
    """Pravděpodobnost AKI podmíněná chronickými diagnózami zprávy.

    Returns:
        P(AKI | chronické diagnózy) pro každou kombinaci chronických diagnóz.
    """
    totals: dict[LabelSet, float] = {}
    acute: dict[LabelSet, float] = {}
    for cell, probability in joint.items():
        chronic = frozenset(d for d in cell if d.chronic)
        totals[chronic] = totals.get(chronic, 0.0) + probability
        if Diagnosis.AKI in cell:
            acute[chronic] = acute.get(chronic, 0.0) + probability
    return {
        chronic: acute.get(chronic, 0.0) / total for chronic, total in totals.items()
    }


type _Table = tuple[tuple[Callable[[LabelSet], bool], float], ...]


def _pair_table(model: LabelModel, first: Diagnosis, second: Diagnosis) -> _Table:
    both = model.co_occurrence(first, second)
    only_first = model.prevalence[first] - both
    only_second = model.prevalence[second] - both
    return (
        (lambda cell: first in cell and second in cell, both),
        (lambda cell: first in cell and second not in cell, only_first),
        (lambda cell: first not in cell and second in cell, only_second),
        (
            lambda cell: first not in cell and second not in cell,
            1 - both - only_first - only_second,
        ),
    )


def _fit(joint: dict[LabelSet, float], table: _Table) -> None:
    for belongs, target in table:
        current = sum(p for cell, p in joint.items() if belongs(cell))
        for cell in [cell for cell in joint if belongs(cell)]:
            joint[cell] *= target / current


def _deviation(joint: Mapping[LabelSet, float], table: _Table) -> float:
    return max(
        abs(sum(p for cell, p in joint.items() if belongs(cell)) - target)
        for belongs, target in table
    )
