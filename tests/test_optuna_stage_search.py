"""Návrhy aktivních větví a shoda obnovené historie."""

from typing import TYPE_CHECKING

import pytest

from mediparse.infrastructure.optuna_search import OptunaStageSearch, stage_sampler
from tests.stage_support import stage_config

if TYPE_CHECKING:
    from mediparse.domain.tuning import Trial


@pytest.mark.parametrize("cut", [3, 5, 8])
def test_restore_matches_uninterrupted(cut: int) -> None:
    """Obnova uvnitř výchozích bodů i v TPE vrací totožné návrhy."""
    config = stage_config()
    search = OptunaStageSearch.create(config.branches, config.seed, 5)
    history: list[tuple[Trial, float]] = []
    for n in range(12):
        if n < 4:
            search.enqueue(config.branches[n].default_params())
        trial = search.ask()
        if n < 4:
            assert trial.params == config.branches[n].default_params()
        branch = next(b for b in config.branches if b.name == trial.params["branch"])
        rep, fam = branch.params_of(trial.params)
        assert branch.representation.space.contains(rep)
        assert branch.family.space.contains(fam)
        search.complete(trial, n / 12)
        history.append((trial, n / 12))
    resumed = OptunaStageSearch.create(config.branches, config.seed, 5)
    resumed.restore(history[:cut])
    for n in range(cut, 12):
        if n < 4:
            resumed.enqueue(config.branches[n].default_params())
        trial = resumed.ask()
        assert trial == history[n][0]
        resumed.complete(trial, history[n][1])


def test_stage_maximizes_objective() -> None:
    """Studie vybírá 0,9 z dokončených hodnot 0,2; 0,9; 0,5."""
    config = stage_config()
    search = OptunaStageSearch.create(config.branches, config.seed, 5)
    for value in (0.2, 0.9, 0.5):
        search.complete(search.ask(), value)
    assert search.best()[1] == pytest.approx(0.9)


def test_stage_never_prunes_after_startup() -> None:
    """Ani výrazný propad po pěti pokusech není důvodem k prořezání."""
    config = stage_config()
    search = OptunaStageSearch.create(config.branches, config.seed, 5)
    for _ in range(5):
        search.complete(search.ask(), 1.0)
    trial = search.ask()
    assert search.should_prune(trial, 0.0, 0) is False


def test_sampler_description_uses_actual_options() -> None:
    """Manifest pojmenuje přesné nastavení TPE včetně group."""
    assert stage_sampler(20) == (
        "optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, group=True)"
    )
