"""Podmíněný prostor ladění stupně a záznamy jeho pokusů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from mediparse.domain.model_input import ModelInput
from mediparse.domain.representation import (
    ClassifierFamily,
    EncoderSpec,
    Representation,
    TfidfRepresentation,
    TokensSpec,
    Word2VecRepresentation,
    accepts,
)
from mediparse.domain.seed import Seed
from mediparse.domain.tuning import (
    Parameter,
    ParamValue,
    SearchSpace,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.split import SplitConfig
    from mediparse.domain.tuning import MaxF1

BRANCH_KEY: Final = "branch"


def _tuned(
    representation: Representation,
) -> tuple[tuple[Parameter, ...], Mapping[str, ParamValue]]:
    match representation:
        case TfidfRepresentation() | Word2VecRepresentation():
            return representation.space.params, representation.default
        case EncoderSpec() | TokensSpec():
            return (), {}
        case _:
            assert_never(representation)


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

    representation: Representation
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
                (self.representation.name, _tuned(self.representation)[0]),
                (self.family.family.value, self.family.space.params),
            )
            for param in space
        )

    def default_params(self) -> dict[str, ParamValue]:
        """Sestaví výchozí bod včetně volby větve.

        Returns:
            Úplné parametry pokusu.
        """
        params: dict[str, ParamValue] = {BRANCH_KEY: self.name}
        for prefix, default in (
            (self.representation.name, _tuned(self.representation)[1]),
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
                for p in _tuned(self.representation)[0]
            },
            {
                p.name: trial_params[f"{self.family.family.value}.{p.name}"]
                for p in self.family.space.params
            },
        )


class StageLsa(BaseModel):
    """Rozměry a paměťový strop reprezentace LSA."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    components: PositiveInt
    max_features: PositiveInt

    @model_validator(mode="after")
    def _cap_exceeds_components(self) -> Self:
        if self.max_features <= self.components:
            msg = "Strop slovníku LSA musí převyšovat počet komponent"
            raise ValueError(msg)
        return self


class StageTuningConfig(BaseModel):
    """Rozpočet jednoho CASH ladění na vývojové části."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    seed: Seed
    subsample_parts: Annotated[int, Field(ge=1)]
    folds: Annotated[int, Field(ge=2)]
    trials: PositiveInt
    random_startup_trials: Annotated[int, Field(ge=0)]
    lsa: StageLsa | None = None
    input: ModelInput
    representations: Annotated[tuple[Representation, ...], Field(min_length=1)]
    families: Annotated[tuple[FamilySpace, ...], Field(min_length=1)]

    @property
    def branches(self) -> tuple[Branch, ...]:
        """Kompatibilní dvojice reprezentací a rodin v pořadí configu."""
        return tuple(
            Branch(rep, fam)
            for rep in self.representations
            for fam in self.families
            if accepts(rep, fam.family)
        )

    @property
    def encoders(self) -> tuple[EncoderSpec, ...]:
        """Encodery použité v ladění a zapsané do provenance."""
        return tuple(
            rep for rep in self.representations if isinstance(rep, EncoderSpec)
        )

    def branch(self, params: Mapping[str, ParamValue]) -> Branch:
        """Vyhledá větev aktivní v parametrech pokusu.

        Returns:
            Deklarovaná dvojice reprezentace a rodiny.
        """
        return next(
            branch for branch in self.branches if branch.name == params[BRANCH_KEY]
        )

    @property
    def startup_trials(self) -> int:
        """Počet výchozích a náhodných pokusů před TPE."""
        return len(self.branches) + self.random_startup_trials

    @model_validator(mode="after")
    def _valid_stage(self) -> Self:
        if (self.lsa is not None) != any(
            f.family is ClassifierFamily.LIGHTGBM_LSA for f in self.families
        ):
            msg = "LSA musí být zadané právě s rodinou lightgbm_lsa."
            raise ValueError(msg)
        if len({r.name for r in self.representations}) != len(self.representations):
            msg = "Jména reprezentací musí být jedinečná."
            raise ValueError(msg)
        if len({f.family for f in self.families}) != len(self.families):
            msg = "Rodiny musí být jedinečné."
            raise ValueError(msg)
        for family in self.families:
            if not any(accepts(rep, family.family) for rep in self.representations):
                msg = f"Rodina {family.family} nepatří do žádné větve."
                raise ValueError(msg)
        for rep in self.representations:
            if not any(accepts(rep, family.family) for family in self.families):
                msg = f"Reprezentace {rep.name} nepatří do žádné větve."
                raise ValueError(msg)
        if self.trials < self.startup_trials:
            msg = "Rozpočet musí pokrýt výchozí i náhodné pokusy."
            raise ValueError(msg)
        if self.seed + self.trials >= 2**32:
            msg = "Seed spolu s rozpočtem překračuje rozsah sampleru."
            raise ValueError(msg)
        return self


def branch_parts(name: str) -> tuple[str, ClassifierFamily]:
    """Rozloží doménovou identitu větve pro tabulkový výstup.

    Returns:
        Jméno reprezentace a rodina klasifikátoru.
    """
    representation, family = name.split("/")
    return representation, ClassifierFamily(family)


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
    n_features: int

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
