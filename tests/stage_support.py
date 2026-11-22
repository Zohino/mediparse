"""Malé prostory stupně pro syntetická ověření."""

import json
from dataclasses import dataclass

from mediparse.domain.stage_tuning import StageTuningConfig


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
        params: list[dict[str, object]] = [
            {"name": key, "kind": "categorical", "choices": [value]}
            for key, value in default.items()
        ]
        params[3] = {
            "name": "max_df",
            "kind": "float",
            "low": 0.5,
            "high": 1.0,
            "log": False,
        }
        if name == "r1":
            params[2] = {
                "name": "min_df",
                "kind": "int",
                "low": 1,
                "high": 2,
                "log": True,
            }
        representations.append({
            "name": name,
            "text_preprocessing": "dictionary" if name == "r1" else "none",
            "analyzer": "word",
            "space": {"params": params},
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
                "strip_diagnostic_sections": False,
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


@dataclass
class StepClock:
    """Hodiny s konstantním krokem pro bajtově shodné výstupy."""

    value: float = 0.0

    def now(self) -> float:
        """Posune čas o sekundu.

        Returns:
            Nový stav hodin.
        """
        self.value += 1.0
        return self.value


def token_stage_config() -> StageTuningConfig:
    """Sestaví malý prostor fastTextu bez parametrů reprezentace.

    Returns:
        Tokenová větev s pevným levným fitem.
    """
    default = {"lr": 0.5, "epoch": 1, "word_ngrams": 2, "dim": 2}
    return stage_config(
        representations=[{"name": "tok", "text_preprocessing": "dictionary"}],
        families=[
            {
                "family": "fasttext",
                "space": {
                    "params": [
                        {"kind": "categorical", "name": name, "choices": [value]}
                        for name, value in default.items()
                    ]
                },
                "default": default,
            }
        ],
    )


def word2vec_stage_config() -> StageTuningConfig:
    """Sestaví levnou větev word2vec pro syntetická ověření.

    Returns:
        Word2vec s jednovláknovým tréninkem a logistickou regresí.
    """
    default = {
        "vector_size": 7,
        "window": 2,
        "min_count": 2,
        "sg": 0,
        "epochs": 5,
        "negative": 5,
        "sample": 0.001,
    }
    params: list[dict[str, object]] = [
        {"kind": "categorical", "name": name, "choices": [value]}
        for name, value in default.items()
        if name != "sample"
    ]
    params.append({
        "kind": "float",
        "name": "sample",
        "low": 0.001,
        "high": 0.002,
        "log": False,
    })
    return stage_config(
        representations=[
            {
                "name": "w2v",
                "text_preprocessing": "dictionary",
                "workers": 1,
                "space": {"params": params},
                "default": default,
            }
        ],
        families=stage_config().model_dump(mode="json")["families"][:1],
    )
