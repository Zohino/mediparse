"""Doménové invarianty podmíněného prostoru a vítěze."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from mediparse.domain import stage_tuning as stage_domain
from mediparse.domain.stage_tuning import (
    StageTrialRecord,
    StageTuningConfig,
    best_trial,
)
from tests.stage_support import stage_config


def test_branches_and_namespaces() -> None:
    """Reprezentace tvoří vnější smyčku a rodiny sdílejí jmenný prostor."""
    config = stage_config()
    assert config.startup_trials == 5
    assert [b.name for b in config.branches] == [
        "r1/logreg",
        "r1/complement_nb",
        "r2/logreg",
        "r2/complement_nb",
    ]
    for branch in config.branches:
        params = branch.default_params()
        rep, fam = branch.params_of(params)
        assert rep == branch.representation.default
        assert fam == branch.family.default
        assert {p.name for p in branch.prefixed_space()} == params.keys() - {"branch"}
        assert params["branch"] == branch.name


def test_winner_tie_uses_earlier_trial() -> None:
    """Hodnoty 0,7; 0,9; 0,9 vyberou pokus 1."""
    records = [
        StageTrialRecord(n, {}, default=False, value=value, converged=True, seconds=1.0)
        for n, value in enumerate([0.7, 0.9, 0.9])
    ]
    assert best_trial(records).number == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {("families", 0, "default", "C"): 10000.0},
            "Výchozí parametry rodiny",
            id="family-default-outside",
        ),
        pytest.param(
            {("representations", 0, "default", "max_df"): 1},
            "Výchozí parametry reprezentace",
            id="max-df-int-default",
        ),
        pytest.param({("trials",): 4}, "Rozpočet", id="insufficient-budget"),
        pytest.param(
            {
                ("representations", 0, "space", "params", 0): {
                    "kind": "int",
                    "name": "ngram_min",
                    "low": 1,
                    "high": 3,
                    "log": False,
                }
            },
            "Maximum ngram_min",
            id="ngram-int-order",
        ),
        pytest.param(
            {
                ("representations", 0, "space", "params", 2): {
                    "kind": "float",
                    "name": "min_df",
                    "low": 0.1,
                    "high": 1.0,
                    "log": False,
                },
                ("representations", 0, "default", "min_df"): 1.0,
            },
            "min_df vyžaduje celočíselný",
            id="min-df-float-space",
        ),
        pytest.param(
            {("families", 0, "family"): "fasttext"},
            "Rodina fasttext",
            id="incompatible-family",
        ),
        pytest.param(
            {("representations", 1, "name"): "r1"},
            "Jména reprezentací",
            id="duplicate-representation",
        ),
        pytest.param(
            {("seed",): 4294967286, ("trials",): 60}, "Seed spolu", id="seed-overflow"
        ),
        pytest.param(
            {("representations", 0, "space", "params", 0, "name"): "unknown"},
            "právě pět parametrů",
            id="invalid-parameter-names",
        ),
        pytest.param(
            {
                ("representations", 0, "space", "params", 3): {
                    "kind": "categorical",
                    "name": "max_df",
                    "choices": [1.0],
                }
            },
            "max_df vyžaduje float",
            id="max-df-category",
        ),
        pytest.param(
            {
                ("representations", 0, "space", "params", 4): {
                    "kind": "categorical",
                    "name": "sublinear_tf",
                    "choices": [1],
                }
            },
            "sublinear_tf bool",
            id="sublinear-int",
        ),
        pytest.param(
            {
                ("families", 1): {
                    "family": "logreg",
                    "space": {
                        "params": [
                            {"kind": "categorical", "name": "C", "choices": [1.0]}
                        ]
                    },
                    "default": {"C": 1.0},
                }
            },
            "Rodiny musí být jedinečné",
            id="duplicate-family",
        ),
        pytest.param(
            {
                ("representations", 0, "space", "params", 0, "choices"): [1, 3],
                ("representations", 0, "space", "params", 1, "choices"): [2, 4],
            },
            "Maximum ngram_min",
            id="ngram-category-order",
        ),
    ],
)
def test_invalid_config(
    changes: dict[tuple[str | int, ...], object], message: str
) -> None:
    """Každou neplatnou úpravu odmítne její vlastní validační pravidlo."""
    data = stage_config().model_dump(mode="json")
    for path, value in changes.items():
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    with pytest.raises(ValidationError, match=message):
        StageTuningConfig.model_validate_json(json.dumps(data))


def test_branch_identity_is_resolved_in_domain() -> None:
    """Doména vrátí větev i její části z identity uložené v pokusu."""
    config = stage_config()
    for branch in config.branches:
        params = branch.default_params()
        assert params[stage_domain.BRANCH_KEY] == branch.name
        assert config.branch(params) == branch
        assert stage_domain.branch_parts(branch.name) == (
            branch.representation.name,
            branch.family.family,
        )
