"""Malé prostory stupně pro syntetická ověření."""

import json

from mediparse.domain.stage_tuning import StageTuningConfig
from tests.model_support import model_input


def stage_config(**changes: object) -> StageTuningConfig:
    """Sestaví dvě reprezentace a dvě rodiny.

    Returns:
        Šest pokusů s jedním náhodným startem.
    """
    representations = []
    for name in ("r1", "r2"):
        default = {
            "ngram_min": 1,
            "ngram_max": 2,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
        }
        representations.append({
            "name": name,
            "analyzer": "word",
            "space": {
                "params": [
                    {
                        "name": key,
                        "kind": "float",
                        "low": 0.5,
                        "high": 1.0,
                        "log": False,
                    }
                    if key == "max_df"
                    else {"name": key, "kind": "categorical", "choices": [value]}
                    for key, value in default.items()
                ]
            },
            "default": default,
        })
    return StageTuningConfig.model_validate_json(
        json.dumps(
            {
                "seed": 7,
                "subsample_parts": 1,
                "folds": 2,
                "trials": 6,
                "random_startup_trials": 1,
                "lsa_components": 2,
                "input": model_input(),
                "representations": representations,
                "families": [
                    {
                        "family": family,
                        "space": {
                            "params": [
                                {
                                    "kind": "float",
                                    "name": param,
                                    "low": 0.001,
                                    "high": 1000.0,
                                    "log": True,
                                }
                            ]
                        },
                        "default": {param: 1.0},
                    }
                    for family, param in (("logreg", "C"), ("complement_nb", "alpha"))
                ],
            }
            | changes
        )
    )
