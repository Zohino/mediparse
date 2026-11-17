"""Optuna jako strategie návrhů a prořezávání, se sqlite pro jednotlivý běh."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.exceptions import DuplicatedStudyError
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from sqlalchemy.pool import NullPool

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import CategoricalParam, FloatParam, IntParam, Trial

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mediparse.domain.seed import Seed
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
class OptunaSearch:
    """Převádí doménové pokusy na ask-and-tell Optuny bez řízení smyčky."""

    study: optuna.Study
    distributions: dict[str, BaseDistribution]
    # Port nese jen doménové číslo pokusu, report proto potřebuje původní objekt Optuny.
    pending: dict[int, optuna.Trial] = field(default_factory=dict)

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


def _distribution(param: FloatParam | IntParam | CategoricalParam) -> BaseDistribution:
    if isinstance(param, FloatParam):
        return FloatDistribution(param.low, param.high, log=param.log)
    if isinstance(param, IntParam):
        return IntDistribution(param.low, param.high, log=param.log)
    return CategoricalDistribution(param.choices)
