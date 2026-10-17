"""Doména tréninku smoketestu: neplatný config."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.smoketest_training import TrainingConfig
from tests.model_support import training_config

FOLDS = 2
VALID = training_config(row_id="smoke", diagnosis="diabetes", folds=FOLDS)


def test_valid_config_is_parsed() -> None:
    """Platný JSON dá config s výčtem diagnózy."""
    config = TrainingConfig.model_validate_json(json.dumps(VALID))

    assert config.row_id == "smoke"
    assert config.diagnosis is Diagnosis.DIABETES
    assert config.folds == FOLDS


def test_config_without_row_id_is_rejected() -> None:
    """Config bez row_id neprojde."""
    without = {key: value for key, value in VALID.items() if key != "row_id"}

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate_json(json.dumps(without))


@pytest.mark.parametrize(
    "override",
    [
        {"folds": 1},
        {"regularization": 0.0},
        {"regularization": -1.0},
        {"diagnosis": "asthma"},
        {"unexpected": 1},
        {"row_id": ""},
        {"row_id": "Smoke"},
        {"row_id": "smoke test"},
    ],
)
def test_invalid_config_is_rejected(override: dict[str, object]) -> None:
    """Málo foldů, nekladné C, neznámá diagnóza, extra klíč i neplatný row_id config odmítnou."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate_json(json.dumps({**VALID, **override}))
