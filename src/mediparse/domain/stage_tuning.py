"""Podmíněný prostor ladění stupně a záznamy jeho pokusů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from mediparse.domain.model_input import ModelInput
from mediparse.domain.screening import ClassifierFamily
from mediparse.domain.seed import Seed
from mediparse.domain.tuning import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParamValue,
    SearchSpace,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.split import SplitConfig
    from mediparse.domain.tuning import MaxF1

type Parameter = FloatParam | IntParam | CategoricalParam


def _int_choices(param: Parameter) -> tuple[int, int]:
    if isinstance(param, IntParam):
        return param.low, param.high
    if isinstance(param, CategoricalParam):
        choices = tuple(v for v in param.choices if type(v) is int)
        if len(choices) == len(param.choices):
            return min(choices), max(choices)
    msg = f"Parametr {param.name} vyžaduje celočíselný prostor."
    raise ValueError(msg)


class TfidfRepresentation(BaseModel):
    """Pojmenovaná reprezentace se striktním prostorem TF-IDF."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, pattern=r"^[^./]+$")]
    analyzer: Literal["word", "char_wb"]
    space: SearchSpace
    default: dict[str, ParamValue]

    @model_validator(mode="after")
    def _valid_space(self) -> Self:
        params = {p.name: p for p in self.space.params}
        if params.keys() != {
            "ngram_min",
            "ngram_max",
            "min_df",
            "max_df",
            "sublinear_tf",
        }:
            msg = "Prostor TF-IDF musí obsahovat právě pět parametrů reprezentace."
            raise ValueError(msg)
        ngram_min = _int_choices(params["ngram_min"])
        ngram_max = _int_choices(params["ngram_max"])
        _int_choices(params["min_df"])
        if ngram_min[1] > ngram_max[0]:
            msg = "Maximum ngram_min nesmí překročit minimum ngram_max."
            raise ValueError(msg)
        flag = params["sublinear_tf"]
        if not isinstance(params["max_df"], FloatParam) or not (
            isinstance(flag, CategoricalParam)
            and all(type(v) is bool for v in flag.choices)
        ):
            msg = "max_df vyžaduje float a sublinear_tf bool kategorie."
            raise ValueError(msg)
        if not self.space.contains(self.default):
            msg = "Výchozí parametry reprezentace neleží v prostoru."
            raise ValueError(msg)
        return self


class FamilySpace(BaseModel):
    """Prostor jedné rodiny s výchozím bodem."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    family: ClassifierFamily
    space: SearchSpace
    default: dict[str, ParamValue]

    @model_validator(mode="after")
    def _default_in_space(self) -> Self:
        if not self.space.contains(self.default):
            msg = "Výchozí parametry rodiny neleží v prostoru."
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class Branch:
    """Platná dvojice reprezentace a rodiny se společnými jmennými prostory."""

    representation: TfidfRepresentation
    family: FamilySpace

    @property
    def name(self) -> str:
        """Jednoznačné jméno dvojice."""
        return f"{self.representation.name}/{self.family.family.value}"

    def prefixed_space(self) -> tuple[Parameter, ...]:
        """Přidá předpony reprezentace a rodiny k parametrům.

        Returns:
            Parametry ve stabilním pořadí configu.
        """
        return tuple(
            param.model_copy(update={"name": f"{prefix}.{param.name}"})
            for prefix, space in (
                (self.representation.name, self.representation.space),
                (self.family.family.value, self.family.space),
            )
            for param in space.params
        )

    def default_params(self) -> dict[str, ParamValue]:
        """Sestaví výchozí bod včetně volby větve.

        Returns:
            Úplné parametry pokusu.
        """
        params: dict[str, ParamValue] = {"branch": self.name}
        for prefix, default in (
            (self.representation.name, self.representation.default),
            (self.family.family.value, self.family.default),
        ):
            params.update({
                f"{prefix}.{name}": value for name, value in default.items()
            })
        return params

    def params_of(
        self, trial_params: Mapping[str, ParamValue]
    ) -> tuple[dict[str, ParamValue], dict[str, ParamValue]]:
        """Oddělí parametry reprezentace a rodiny.

        Returns:
            Dvojice slovníků bez předpon.
        """
        return (
            {
                p.name: trial_params[f"{self.representation.name}.{p.name}"]
                for p in self.representation.space.params
            },
            {
                p.name: trial_params[f"{self.family.family.value}.{p.name}"]
                for p in self.family.space.params
            },
        )


class StageTuningConfig(BaseModel):
    """Rozpočet jednoho CASH ladění na vývojové části."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    seed: Seed
    subsample_parts: Annotated[int, Field(ge=1)]
    folds: Annotated[int, Field(ge=2)]
    trials: PositiveInt
    random_startup_trials: Annotated[int, Field(ge=0)]
    lsa_components: PositiveInt
    input: ModelInput
    representations: Annotated[tuple[TfidfRepresentation, ...], Field(min_length=1)]
    families: Annotated[tuple[FamilySpace, ...], Field(min_length=1)]

    @property
    def branches(self) -> tuple[Branch, ...]:
        """Součin reprezentací a rodin v pořadí configu."""
        return tuple(
            Branch(rep, fam) for rep in self.representations for fam in self.families
        )

    @property
    def startup_trials(self) -> int:
        """Počet výchozích a náhodných pokusů před TPE."""
        return len(self.branches) + self.random_startup_trials

    @model_validator(mode="after")
    def _valid_stage(self) -> Self:
        if len({r.name for r in self.representations}) != len(self.representations):
            msg = "Jména reprezentací musí být jedinečná."
            raise ValueError(msg)
        if len({f.family for f in self.families}) != len(self.families):
            msg = "Rodiny musí být jedinečné."
            raise ValueError(msg)
        if any(f.family is ClassifierFamily.FASTTEXT for f in self.families):
            msg = "Rodina fasttext nepracuje nad reprezentací TF-IDF."
            raise ValueError(msg)
        if self.trials < self.startup_trials:
            msg = "Rozpočet musí pokrýt výchozí i náhodné pokusy."
            raise ValueError(msg)
        if self.seed + self.trials >= 2**32:
            msg = "Seed spolu s rozpočtem překračuje rozsah sampleru."
            raise ValueError(msg)
        return self


@dataclass(frozen=True, kw_only=True)
class StageFoldResult:
    """Skóre a optimální práh jedné diagnózy validačního foldu."""

    fold: int
    diagnosis: Diagnosis
    note_ids: tuple[str, ...]
    labels: tuple[bool, ...]
    scores: tuple[float, ...]
    max_f1: MaxF1
    converged: bool

    def __post_init__(self) -> None:
        """Ověří délky sloupců buňky.

        Raises:
            ValueError: Identity, labely a skóre nemají stejné délky.
        """
        if not (len(self.note_ids) == len(self.labels) == len(self.scores)):
            msg = "Identity, labely a skóre musí mít stejné délky."
            raise ValueError(msg)


@dataclass(frozen=True)
class StageTrialRecord:
    """Parametry, cíl a cena dokončeného pokusu."""

    number: int
    params: Mapping[str, ParamValue]
    default: bool
    value: float
    converged: bool
    seconds: float


def best_trial(records: Sequence[StageTrialRecord]) -> StageTrialRecord:
    """Vybere nejvyšší cíl a při stejném cíli dřívější pokus.

    Returns:
        Vítězný záznam.
    """
    return max(records, key=lambda record: (record.value, -record.number))


@dataclass(frozen=True)
class StageTuningReport:
    """Vítěz, rozpočet a velikosti vzorku s provenance splitu."""

    best: StageTrialRecord
    completed_trials: int
    seconds: float
    nonconverged_trials: int
    sample_notes: int
    sample_patients: int
    development_notes: int
    development_patients: int
    split_config: SplitConfig
    split_sha256: str
