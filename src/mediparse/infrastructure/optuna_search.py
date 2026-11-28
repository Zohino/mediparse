"""Studie stupně v paměti obnovovaná z checkpointu."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, TypedDict

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.trial import create_trial

from mediparse.domain.stage_tuning import BRANCH_KEY
from mediparse.domain.tuning import CategoricalParam, FloatParam, IntParam, Trial

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue


def _distribution(param: FloatParam | IntParam | CategoricalParam) -> BaseDistribution:
    if isinstance(param, FloatParam):
        return FloatDistribution(param.low, param.high, log=param.log)
    if isinstance(param, IntParam):
        return IntDistribution(param.low, param.high, log=param.log)
    return CategoricalDistribution(param.choices)


STAGE_PRUNER: Final = "optuna.pruners.NopPruner"


class _StageSamplerOptions(TypedDict):
    n_startup_trials: int
    multivariate: bool
    group: bool
    constant_liar: bool


def _stage_sampler_options(startup_trials: int) -> _StageSamplerOptions:
    # Připnutí výchozí volby chrání provenance před změnou verze Optuny.
    return {
        "n_startup_trials": startup_trials,
        "multivariate": True,
        "group": True,
        "constant_liar": True,
    }


def stage_sampler(startup_trials: int) -> str:
    """Popíše parametry sampleru pro provenance.

    Returns:
        Jednoznačný popis strategie.
    """
    options = ", ".join(
        f"{name}={value}"
        for name, value in _stage_sampler_options(startup_trials).items()
    )
    return f"optuna.samplers.TPESampler({options})"


@dataclass(frozen=True)
class OptunaStageSearch:
    """Podmíněné CASH návrhy s obnovou z dokončených pokusů."""

    study: optuna.Study
    config: StageTuningConfig
    # tell potřebuje původní objekt Optuny, port zná jen číslo.
    pending: dict[int, optuna.Trial] = field(default_factory=dict, kw_only=True)

    def complete(self, trial: Trial, value: float) -> None:
        """Uzavře dokončený pokus s cílovou hodnotou."""
        self.study.tell(self.pending.pop(trial.number), value)

    @classmethod
    def create(cls, config: StageTuningConfig) -> OptunaStageSearch:
        """Vytvoří prázdnou studii v paměti bez prořezávání.

        Returns:
            Adaptér nad deklarovanými větvemi.
        """
        return cls(
            optuna.create_study(direction="maximize", pruner=NopPruner()),
            config,
        )

    def enqueue(self, params: Mapping[str, ParamValue]) -> None:
        """Vloží výchozí bod pro následující ask."""
        self.study.enqueue_trial(dict(params))

    def ask(self) -> Trial:
        """Navrhne větev a její aktivní parametry.

        Returns:
            Pokus se jmennými prostory reprezentace a rodiny.
        """
        # Nový seed podle historie dává po pádu stejný proud návrhů jako běh vcelku.
        # group je experimentální API; filtr patří jen ke konstrukci sampleru.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            self.study.sampler = TPESampler(
                seed=self.config.seed + len(self.study.trials),
                **_stage_sampler_options(self.config.sampler_startup_trials),
            )
        trial = self.study.ask()
        trial.suggest_categorical(
            BRANCH_KEY, tuple(b.name for b in self.config.branches)
        )
        for param in self.config.branch(trial.params).prefixed_space():
            match param:
                case FloatParam():
                    trial.suggest_float(
                        param.name, param.low, param.high, log=param.log
                    )
                case IntParam():
                    trial.suggest_int(param.name, param.low, param.high, log=param.log)
                case CategoricalParam():
                    trial.suggest_categorical(param.name, param.choices)
        self.pending[trial.number] = trial
        return Trial(trial.number, dict(trial.params))

    def restore(self, trials: Sequence[tuple[Trial, float]]) -> None:
        """Znovu postaví historii studie z checkpointu."""
        for trial, value in trials:
            branch = self.config.branch(trial.params)
            distributions = {
                BRANCH_KEY: CategoricalDistribution(
                    tuple(b.name for b in self.config.branches)
                ),
                **{p.name: _distribution(p) for p in branch.prefixed_space()},
            }
            self.study.add_trial(
                create_trial(
                    params=dict(trial.params),
                    distributions=distributions,
                    value=value,
                )
            )
