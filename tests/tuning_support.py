"""Syntetické konfigurace ladění se stejnými parametry jako produkční prostor."""

from __future__ import annotations

import json

from mediparse.domain.matrix import MatrixRow
from mediparse.domain.tuning import ParamValue, TuningConfig
from tests.model_support import model_input

SVM_PARAMS: dict[str, ParamValue] = {
    "analyzer": "word",
    "ngram_max": 2,
    "c": 0.5,
    "class_weight": "balanced",
    "min_df": 1,
    "max_df": 1.0,
    "sublinear_tf": True,
}


def tuning_config(**overrides: object) -> TuningConfig:
    """Sestaví malý prostor, který nemůže vyprázdnit slovník syntetického korpusu.

    Returns:
        Striktní config s dvěma vnitřními foldy a seedem 7.
    """
    return TuningConfig.model_validate_json(
        json.dumps(
            {
                "row_id": "en-svm",
                "fold": 0,
                "seed": 7,
                "inner_folds": 2,
                "pruner_startup_trials": 1,
                "space": {
                    "params": [
                        {
                            "kind": "float",
                            "name": "c",
                            "low": 0.1,
                            "high": 3.0,
                            "log": True,
                        },
                        *[
                            {"kind": "categorical", "name": name, "choices": [value]}
                            for name, value in SVM_PARAMS.items()
                            if name != "c"
                        ],
                    ]
                },
            }
            | overrides
        )
    )


def matrix_row(**overrides: object) -> MatrixRow:
    """Sestaví jeden řádek matice s ručně určeným rozpočtem tří pokusů.

    Returns:
        Řádek SVM bez úprav textu, přepsaný dodanými poli.
    """
    return MatrixRow.model_validate_json(
        json.dumps(
            {
                "id": "en-svm",
                "language": "en",
                "model": "svm",
                "input": model_input(),
                "tuning_trials": 3,
            }
            | overrides
        )
    )
