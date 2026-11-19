"""Doménové invarianty podmíněného prostoru a vítěze."""

import json

import pytest
from pydantic import ValidationError

from mediparse.domain.stage_tuning import (
    StageTrialRecord,
    StageTuningConfig,
    best_trial,
)
from tests.stage_support import stage_config


def test_branches_and_namespaces() -> None:
    """Reprezentace tvoří vnější smyčku a rodiny sdílejí jmenný prostor."""
    config = stage_config()
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


@pytest.mark.parametrize("case", range(8))
def test_invalid_config(case: int) -> None:
    """Osm odmítnutí podle nezávislých příkladů briefu E8."""
    data = stage_config().model_dump(mode="json")
    match case:
        case 0:
            data["families"][0]["default"]["C"] = 1e4
        case 1:
            data["representations"][0]["default"]["max_df"] = 1
        case 2:
            data["trials"] = 4
        case 3:
            params = data["representations"][0]["space"]["params"]
            params[0] = {
                "kind": "int",
                "name": "ngram_min",
                "low": 1,
                "high": 3,
                "log": False,
            }
        case 4:
            params = data["representations"][0]["space"]["params"]
            params[2] = {
                "kind": "float",
                "name": "min_df",
                "low": 0.1,
                "high": 1.0,
                "log": False,
            }
        case 5:
            data["families"][0]["family"] = "fasttext"
        case 6:
            data["representations"][1]["name"] = "r1"
        case 7:
            data.update(seed=2**32 - 10, trials=60)
    with pytest.raises(ValidationError):
        StageTuningConfig.model_validate_json(json.dumps(data))
