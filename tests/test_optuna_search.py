"""Převod doménového prostoru a životní cyklus skutečné studie Optuny."""

from __future__ import annotations

import gc
import json
import warnings
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import SearchSpace
from mediparse.infrastructure.optuna_search import OptunaSearch
from mediparse.infrastructure.tuning_config import load_tuning_config
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


@given(
    low=st.integers(1, 100),
    width=st.integers(1, 100),
    log=st.booleans(),
    choice=st.sampled_from([None, True, 3, 0.25, "word"]),
)
def test_ask_preserves_space_types_bounds_and_log(
    low: int, width: int, *, log: bool, choice: object
) -> None:
    """Vygenerovaný prostor má samostatné int, float i kategorii včetně null."""
    high = low + width
    space = SearchSpace.model_validate_json(
        json.dumps({
            "params": [
                {
                    "kind": "float",
                    "name": "real",
                    "low": float(low),
                    "high": float(high),
                    "log": log,
                },
                {
                    "kind": "int",
                    "name": "integer",
                    "low": low,
                    "high": high,
                    "log": log,
                },
                {"kind": "categorical", "name": "category", "choices": [choice]},
            ]
        })
    )
    search = OptunaSearch.create(space, 0, 1, None)
    trial = search.ask()
    assert type(trial.params["real"]) is float
    assert type(trial.params["integer"]) is int
    assert low <= trial.params["real"] <= high
    assert low <= trial.params["integer"] <= high
    assert trial.params["category"] == choice
    assert type(trial.params["category"]) is type(choice)
    distributions = search.study.trials[0].distributions
    real, integer = distributions["real"], distributions["integer"]
    assert isinstance(real, FloatDistribution)
    assert isinstance(integer, IntDistribution)
    assert real.log is log
    assert integer.log is log
    search.complete(trial, 0.5)
    assert search.best() == (trial.params, 0.5)


def test_sqlite_lifecycle_counts_pruned_and_refuses_existing_study(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Jeden COMPLETE a jeden PRUNED jsou dva pokusy, další create studii nepřevezme."""
    space = SearchSpace.model_validate_json(
        '{"params":[{"kind":"categorical","name":"fixed","choices":[null]}]}'
    )
    storage = tmp_path / "nested/tuning.sqlite3"
    search = OptunaSearch.create(space, 7, 1, storage)
    with caplog.at_level("INFO", logger="mediparse.infrastructure.optuna_search"):
        first = search.ask()
        assert not search.should_prune(first, 1.0, 0)
        search.complete(first, 1.0)
        second = search.ask()
        assert search.should_prune(second, 0.0, 0)
        search.prune(second)
    assert [trial.state for trial in search.study.trials] == [
        TrialState.COMPLETE,
        TrialState.PRUNED,
    ]
    assert search.best() == ({"fixed": None}, 1.0)
    assert [line.split(":", 1)[0] for line in caplog.messages] == [
        "Pokus 0 COMPLETE",
        "Pokus 1 PRUNED",
    ]
    assert all("čas 0:00:" in message for message in caplog.messages)
    assert storage.is_file()
    with pytest.raises(InvalidInputError, match="už existuje"):
        OptunaSearch.create(space, 7, 1, storage)


def test_sqlite_does_not_leave_open_connections_for_gc(tmp_path: Path) -> None:
    """Zánik adaptéru nezanechá sqlite spojení k implicitnímu zavření garbage collectorem."""
    space = SearchSpace.model_validate_json(
        '{"params":[{"kind":"categorical","name":"fixed","choices":[null]}]}'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        search = OptunaSearch.create(space, 0, 1, tmp_path / "tuning.sqlite3")
        search.complete(search.ask(), 1.0)
        del search
        gc.collect()
    assert not [
        warning for warning in caught if issubclass(warning.category, ResourceWarning)
    ]


def test_multivariate_search_with_fixed_categories_emits_no_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Produkční prostor nevydá varování při startu ani po deseti úvodních pokusech."""
    space = load_tuning_config(REPOSITORY / "config/tuning.json").space
    with caplog.at_level("WARNING"):
        search = OptunaSearch.create(space, 7, 5, None)
        for _ in range(20):
            search.complete(search.ask(), 0.5)
    assert not caplog.records
