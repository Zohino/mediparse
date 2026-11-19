"""Optuna jako strategie návrhů a prořezávání, se sqlite pro jednotlivý běh."""

from __future__ import annotations

import logging
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
from optuna.exceptions import DuplicatedStudyError, ExperimentalWarning
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.trial import TrialState, create_trial
from sqlalchemy.pool import NullPool

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.stage_tuning import BRANCH_KEY
from mediparse.domain.tuning import CategoricalParam, FloatParam, IntParam, Trial

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.seed import Seed
    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue, SearchSpace

logger = logging.getLogger(__name__)
TPE_STARTUP_TRIALS: Final = 10
TPE_MULTIVARIATE: Final = True
SAMPLER: Final = (
    f"optuna.samplers.TPESampler(n_startup_trials={TPE_STARTUP_TRIALS}, "
    f"multivariate={TPE_MULTIVARIATE})"
)
PRUNER: Final = "optuna.pruners.MedianPruner"


@dataclass(frozen=True)
class OptunaTrials:
    """Společné reportování a uzavírání pokusů Optuny."""

    study: optuna.Study
    # Port zná jen číslo; report potřebuje původní objekt Optuny.
    pending: dict[int, optuna.Trial] = field(default_factory=dict, kw_only=True)

    def should_prune(self, trial: Trial, value: float, step: int) -> bool:
        """Zapíše průběžný cíl a vyhodnotí pruner.

        Returns:
            Zda společné pravidlo doporučuje pokus prořezat.
        """
        pending = self.pending[trial.number]
        pending.report(value, step)
        return pending.should_prune()

    def complete(self, trial: Trial, value: float) -> None:
        """Uzavře dokončený pokus a zaznamená jeho cenu do logu."""
        finished = self.study.tell(self.pending.pop(trial.number), value)
        logger.info(
            "Pokus %d COMPLETE: macro-F1 %.6f, čas %s.",
            trial.number,
            value,
            finished.duration,
        )

    def prune(self, trial: Trial) -> None:
        """Uzavře prořezaný pokus a zaznamená jeho cenu do logu."""
        finished = self.study.tell(
            self.pending.pop(trial.number), state=TrialState.PRUNED
        )
        logger.info("Pokus %d PRUNED: čas %s.", trial.number, finished.duration)

    def best(self) -> tuple[Mapping[str, ParamValue], float]:
        """Vrátí nejlepší dokončený pokus studie.

        Returns:
            Parametry a macro-F1 pro sestavení manifestu.
        """
        return dict(self.study.best_params), self.study.best_value


@dataclass(frozen=True)
class OptunaSearch(OptunaTrials):
    """Převádí doménové pokusy na ask-and-tell Optuny bez řízení smyčky."""

    distributions: dict[str, BaseDistribution]

    @classmethod
    def create(
        cls,
        space: SearchSpace,
        seed: Seed,
        pruner_startup_trials: int,
        storage: Path | None,
    ) -> OptunaSearch:
        """Vytvoří vždy novou studii, existující stejnojmennou odmítne.

        Returns:
            Adaptér s maximalizací cíle a společným samplerem i prunerem.

        Raises:
            InvalidInputError: Studie v dodaném sqlite už existuje.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if storage is not None:
            storage.parent.mkdir(parents=True, exist_ok=True)
        # Pool jinak nechá fyzická spojení sqlite otevřená až do GC; Python 3.13
        # to hlásí jako ResourceWarning. NullPool je zavře po každém použití.
        backend = (
            None
            if storage is None
            else RDBStorage(
                f"sqlite:///{storage.resolve()}", engine_kwargs={"poolclass": NullPool}
            )
        )
        try:
            study = optuna.create_study(
                study_name="model-tuning",
                storage=backend,
                direction="maximize",
                sampler=TPESampler(
                    seed=seed,
                    n_startup_trials=TPE_STARTUP_TRIALS,
                    multivariate=TPE_MULTIVARIATE,
                ),
                pruner=MedianPruner(
                    n_startup_trials=pruner_startup_trials, n_warmup_steps=0
                ),
            )
        except DuplicatedStudyError as error:
            msg = "Studie ladění už existuje; nový běh vyžaduje nové úložiště."
            raise InvalidInputError(msg) from error
        return cls(study, {param.name: _distribution(param) for param in space.params})

    def ask(self) -> Trial:
        """Navrhne další pokus a uchová jeho technologický protějšek.

        Returns:
            Číslo a všechny navržené parametry.
        """
        trial = self.study.ask(self.distributions)
        self.pending[trial.number] = trial
        return Trial(trial.number, dict(trial.params))


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


def _stage_sampler_options(startup_trials: int) -> _StageSamplerOptions:
    return {"n_startup_trials": startup_trials, "multivariate": True, "group": True}


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
class OptunaStageSearch(OptunaTrials):
    """Podmíněné CASH návrhy s obnovou z dokončených pokusů."""

    config: StageTuningConfig

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
                **_stage_sampler_options(self.config.startup_trials),
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
