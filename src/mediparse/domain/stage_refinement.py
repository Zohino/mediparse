"""Rozšíření prostoru a výsledek doladění vítězné větve stupně."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.representation import (
    ClassifierFamily,
    TfidfRepresentation,
    Word2VecRepresentation,
)
from mediparse.domain.stage_tuning import BRANCH_KEY, StageTuningConfig
from mediparse.domain.tuning import CategoricalParam, Parameter, ParamValue, SearchSpace

if TYPE_CHECKING:
    from mediparse.domain.split import SplitConfig
    from mediparse.domain.stage_tuning import StageTrialRecord


class Bounds(BaseModel):
    """Výslovně měněné meze jednoho parametru."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    low: int | float | None = None
    high: int | float | None = None
    choices: tuple[ParamValue, ...] | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if self.low is None and self.high is None and self.choices is None:
            msg = "Rozšíření musí určit alespoň jednu mez."
            raise ValueError(msg)
        return self


class RefinementConfig(BaseModel):
    """Společné podmínky doladění a rozšíření prostorů všech stupňů."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    random_startup_trials: Annotated[int, Field(ge=0)]
    widened: dict[str, Bounds]


def _widen_parameter(param: Parameter, bounds: Bounds) -> Parameter:
    changed = type(param).model_validate(
        param.model_dump() | bounds.model_dump(exclude_none=True)
    )
    if isinstance(param, CategoricalParam):
        original_choices = {(type(value), value) for value in param.choices}
        contains = isinstance(changed, CategoricalParam) and original_choices < {
            (type(value), value) for value in changed.choices
        }
    else:
        contains = (
            not isinstance(changed, CategoricalParam)
            and changed.low <= param.low
            and changed.high >= param.high
        )
    if not contains or changed == param:
        msg = f"Meze {param.name} musí rozšířit původní prostor."
        raise ValueError(msg)
    return changed


def _widen_space(
    space: SearchSpace, prefix: str, refinement: RefinementConfig
) -> SearchSpace:
    changes = {
        key.removeprefix(f"{prefix}."): bounds
        for key, bounds in refinement.widened.items()
        if key.startswith(f"{prefix}.")
    }
    if changes.keys() - space.names:
        msg = f"Neznámé parametry komponenty {prefix}: {sorted(changes.keys() - space.names)}."
        raise ValueError(msg)
    return SearchSpace(
        params=tuple(
            _widen_parameter(param, changes[param.name])
            if param.name in changes
            else param
            for param in space.params
        )
    )


def widen(stage: StageTuningConfig, refinement: RefinementConfig) -> StageTuningConfig:
    """Rozšíří komponenty přítomné ve stupni s validací nových parametrů.

    Returns:
        Nový config, původní předkolo zůstává beze změny.
    """
    representations = tuple(
        type(rep).model_validate(
            rep.model_dump() | {"space": _widen_space(rep.space, rep.name, refinement)}
        )
        if isinstance(rep, (TfidfRepresentation, Word2VecRepresentation))
        else rep
        for rep in stage.representations
    )
    families = tuple(
        type(family).model_validate(
            family.model_dump()
            | {"space": _widen_space(family.space, family.family.value, refinement)}
        )
        for family in stage.families
    )
    return StageTuningConfig.model_validate(
        stage.model_dump() | {"representations": representations, "families": families}
    )


def branch_config(
    stage: StageTuningConfig, branch: str, *, trials: int, random_startup_trials: int
) -> StageTuningConfig:
    """Zúží studii na vítěznou reprezentaci a rodinu.

    Returns:
        Validovaný config jedné větve nad celou vývojovou částí.

    Raises:
        InvalidInputError: Rozpočet nebo parametry větve nejsou platné.
    """
    selected = stage.branch({BRANCH_KEY: branch})
    try:
        return StageTuningConfig.model_validate(
            stage.model_dump()
            | {
                "representations": (selected.representation,),
                "families": (selected.family,),
                "network": stage.network
                if selected.family.family is ClassifierFamily.MULTIRESCNN
                else None,
                "transformer": stage.transformer
                if selected.family.family is ClassifierFamily.TRANSFORMER
                else None,
                "lsa": stage.lsa
                if selected.family.family is ClassifierFamily.LIGHTGBM_LSA
                else None,
                "subsample_parts": 1,
                "trials": trials,
                "random_startup_trials": random_startup_trials,
            }
        )
    except ValidationError as error:
        msg = f"Config větve doladění není platný: {error}"
        raise InvalidInputError(msg) from error


@dataclass(frozen=True)
class StageRefinementReport:
    """Kontrola kandidátů, vítěz doladění a provenance celé vývojové části."""

    study: StageTuningConfig
    checks: tuple[StageTrialRecord, StageTrialRecord]
    best: StageTrialRecord
    completed_trials: int
    nonconverged_trials: int
    seconds: float
    development_notes: int
    development_patients: int
    split_config: SplitConfig
    split_sha256: str
