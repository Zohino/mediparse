"""Prostor a společná metrika ladění všech modelových rodin."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.identity import RowId
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.seed import Seed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.input_note import InputNote
    from mediparse.domain.split import SplitConfig


type ParamValue = str | int | float | bool | None


class FloatParam(BaseModel):
    """Spojitý interval s volitelným logaritmickým vzorkováním."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    kind: Literal["float"]
    name: Annotated[str, Field(min_length=1)]
    low: float
    high: float
    log: bool

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.low >= self.high or (self.log and self.low <= 0):
            msg = "Float interval vyžaduje low < high a pro log low > 0."
            raise ValueError(msg)
        return self


class IntParam(BaseModel):
    """Celočíselný interval včetně mezí."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["int"]
    name: Annotated[str, Field(min_length=1)]
    low: int
    high: int
    log: bool

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.low > self.high or (self.log and self.low < 1):
            msg = "Int interval vyžaduje low ≤ high a pro log low ≥ 1."
            raise ValueError(msg)
        return self


class CategoricalParam(BaseModel):
    """Konečná neprázdná množina hodnot parametru."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    kind: Literal["categorical"]
    name: Annotated[str, Field(min_length=1)]
    choices: Annotated[tuple[ParamValue, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_choices(self) -> Self:
        if len(set(self.choices)) != len(self.choices):
            msg = "Volby kategorického parametru musí být unikátní."
            raise ValueError(msg)
        return self


class SearchSpace(BaseModel):
    """Celá konfigurace modelu včetně parametrů s jedinou pevnou volbou."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    params: Annotated[
        tuple[
            Annotated[
                FloatParam | IntParam | CategoricalParam, Field(discriminator="kind")
            ],
            ...,
        ],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        if len(self.names) != len(self.params):
            msg = "Jména parametrů prostoru musí být unikátní."
            raise ValueError(msg)
        return self

    @property
    def names(self) -> frozenset[str]:
        """Jména, která musí scorer přijímat beze zbytku."""
        return frozenset(param.name for param in self.params)

    def contains(self, params: Mapping[str, ParamValue]) -> bool:
        """Ověří úplnost, typy a příslušnost hodnot do prostoru.

        Returns:
            Zda konfigurace odpovídá všem a právě deklarovaným parametrům.
        """
        return params.keys() == self.names and all(
            _contains(param, params[param.name]) for param in self.params
        )


def _contains(
    param: FloatParam | IntParam | CategoricalParam, value: ParamValue
) -> bool:
    if isinstance(param, CategoricalParam):
        return any(
            type(value) is type(choice) and value == choice for choice in param.choices
        )
    if isinstance(param, IntParam):
        return type(value) is int and param.low <= value <= param.high
    return type(value) is float and isfinite(value) and param.low <= value <= param.high


class TuningConfig(BaseModel):
    """Řádek a fold evaluace, vnitřní CV a strategie hledání."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    row_id: RowId
    fold: Annotated[int, Field(ge=0)]
    seed: Seed
    inner_folds: Annotated[int, Field(ge=2)]
    pruner_startup_trials: Annotated[int, Field(ge=0)]
    space: SearchSpace


@dataclass(frozen=True)
class Trial:
    """Jeden návrh parametrů bez závislosti na knihovně hledání."""

    number: int
    params: Mapping[str, ParamValue]


@dataclass(frozen=True)
class ValidationScores:
    """Skóre v pořadí validačních zpráv pro všech pět diagnóz.

    Attributes:
        features: Počet sloupců reprezentace předaných klasifikátoru,
            u LSA počet sloupců vstupu SVD.
    """

    scores: Mapping[Diagnosis, tuple[float, ...]]
    converged: bool
    features: int


@dataclass(frozen=True)
class MaxF1:
    """Maximum F1 a nejvyšší práh, který ho dosahuje."""

    value: float
    threshold: float

    def predicts(self, score: float) -> bool:
        """Určí pozitivní predikci podle optimálního prahu.

        Returns:
            Zda skóre dosahuje prahu.
        """
        return score >= self.threshold


def max_f1(labels: Sequence[bool], scores: Sequence[float]) -> MaxF1:
    """Najde maximum F1 přes dosažitelné prahy, stejná skóre nedělí.

    Returns:
        Nejvyšší F1 a nejvyšší práh, který ho dosahuje.

    Raises:
        InvalidInputError: Chybí pozitivní label, délky nesedí nebo skóre není konečné.
    """
    if len(labels) != len(scores) or not all(isfinite(score) for score in scores):
        msg = "Labely a konečná skóre musí mít stejnou délku."
        raise InvalidInputError(msg)
    positives = sum(labels)
    if not positives:
        msg = "Max-F1 vyžaduje aspoň jeden pozitivní label."
        raise InvalidInputError(msg)
    true_positive, best = 0, MaxF1(0.0, max(scores))
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    for predicted, (score, label) in enumerate(ordered, start=1):
        true_positive += label
        if predicted == len(ordered) or score != ordered[predicted][0]:
            value = 2 * true_positive / (predicted + positives)
            if value > best.value:
                best = MaxF1(value, score)
    return best


def fold_max_f1(
    notes: Sequence[InputNote], scores: ValidationScores
) -> dict[Diagnosis, MaxF1]:
    """Spočte maximum F1 a práh každé diagnózy validačního foldu.

    Returns:
        Pět maxim s prahy v pořadí diagnóz.

    Raises:
        InvalidInputError: Scorer nedodal přesně pět diagnóz nebo neplatná skóre.
    """
    if scores.scores.keys() != frozenset(Diagnosis):
        msg = "Scorer musí vrátit skóre pro všech pět diagnóz."
        raise InvalidInputError(msg)
    # Volba prahu na validaci je záměrně optimistická a společná všem ramenům;
    # slouží k řazení konfigurací, nikoli k reportování výkonu.
    return {
        diagnosis: max_f1(
            [note.has_label(diagnosis) for note in notes], scores.scores[diagnosis]
        )
        for diagnosis in Diagnosis
    }


def macro_max_f1(f1: Mapping[Diagnosis, MaxF1]) -> float:
    """Spočte průměr pěti optimálních F1 na validaci.

    Returns:
        Společný cíl pro řazení konfigurací.
    """
    return sum(item.value for item in f1.values()) / len(Diagnosis)


@dataclass(frozen=True)
class TuningOutcome:
    """Nejlepší konfigurace a vyčerpání rozpočtu včetně nekonvergence."""

    best: Mapping[str, ParamValue]
    best_macro_f1: float
    completed_trials: int
    pruned_trials: int
    nonconverged_trials: int


@dataclass(frozen=True)
class TuningReport:
    """Agregáty tréninku a provenance původního splitu pro manifest."""

    outcome: TuningOutcome
    train_notes: int
    train_patients: int
    split_config: SplitConfig
    split_sha256: str
