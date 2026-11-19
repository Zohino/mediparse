"""Skutečné fity a překlad chyb klasického stupně."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.application.patient_cv import patient_holdouts
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import sklearn_stage
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.sklearn_stage import SklearnStageScorer
from tests.screening_support import notes
from tests.stage_support import stage_config

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from sklearn.base import BaseEstimator

    from mediparse.domain.screening import ClassifierFamily


def test_logreg_parameters_change_scores(tmp_path: Path) -> None:
    """Silnější a slabší regularizace dávají různá skóre na stejné syntetice."""
    config = stage_config()
    branches = {b.name: b for b in config.branches}
    scorer = SklearnStageScorer(branches, None, 2, 7, tmp_path)
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    params = config.branches[0].default_params()
    low = scorer.score(params | {"logreg.C": 1e-4}, fold)
    high = scorer.score(params | {"logreg.C": 1e2}, fold)
    assert low.scores[Diagnosis.CKD] != high.scores[Diagnosis.CKD]
    with pytest.raises(InvalidInputError, match="slovník"):
        scorer.score(params | {"r1.min_df": 100000}, fold)


def test_lightgbm_receives_lsa_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Estimator LightGBM vidí právě dvě komponenty místo řídkého TF-IDF."""
    config = stage_config()
    data = config.model_dump(mode="json")
    data["families"][0] = {
        "family": "lightgbm_lsa",
        "space": {
            "params": [{"kind": "categorical", "name": "n_estimators", "choices": [10]}]
        },
        "default": {"n_estimators": 10},
    }
    config = type(config).model_validate_json(json.dumps(data))
    observed = []
    original = sklearn_stage.classifier_estimator

    def estimator(family: ClassifierFamily, seed: int, scratch: Path) -> BaseEstimator:
        model = original(family, seed, scratch).set_params()
        fit = model.fit

        def recording_fit(features: np.ndarray, labels: list[bool]) -> BaseEstimator:
            observed.append(features.shape[1])
            return fit(features, labels)

        monkeypatch.setattr(model, "fit", recording_fit)
        return model

    monkeypatch.setattr(sklearn_stage, "classifier_estimator", estimator)
    scorer = SklearnStageScorer(
        {b.name: b for b in config.branches}, None, 2, 7, tmp_path
    )
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    scorer.score(config.branches[0].default_params(), fold)
    assert observed == [2] * 5
