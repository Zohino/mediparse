"""Návrhy aktivních větví a shoda obnovené historie."""

from typing import TYPE_CHECKING

import pytest

from mediparse.infrastructure.optuna_search import OptunaStageSearch
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
