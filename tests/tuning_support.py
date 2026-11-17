"""Syntetické konfigurace ladění se stejnými parametry jako produkční prostor."""

from __future__ import annotations

import json

from mediparse.domain.matrix import MatrixRow
from mediparse.domain.run_manifest import TuningManifest
from mediparse.domain.tuning import ParamValue, TuningConfig
from tests.manifest_support import training_manifest
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


def tuning_manifest(
    *, best: dict[str, ParamValue] | None = None, **overrides: object
) -> TuningManifest:
    """Sestaví syntetický manifest s jednobodovým prostorem pro zadané nejlepší parametry.

    Returns:
        Manifest s ověřitelnými pevnými agregáty a přepsanými poli.
    """
    training = training_manifest()
    parameters = SVM_PARAMS if best is None else best
    fields = {
        "row_id": "row",
        "fold": 0,
        "dataset_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "matrix_sha256": "e" * 64,
        "split_seed": 0,
        "split_folds": 2,
        "seed": 0,
        "tuning_trials": 1,
        "completed_trials": 1,
        "pruned_trials": 0,
        "nonconverged_trials": 0,
        "best_macro_f1": 0.5,
        "best": parameters,
        "sampler": "synthetic",
        "pruner": "synthetic",
        "seconds": 0.0,
        "train_notes": 20,
        "train_patients": 10,
        "diagnostic_strip": None,
        "text_preprocessing": "none",
        "text_normalization": None,
        "environment": training.environment.model_dump(mode="json"),
        "hardware": training.hardware.model_dump(mode="json"),
    } | overrides
    fields["space"] = {
        "params": [
            {"kind": "categorical", "name": name, "choices": [value]}
            for name, value in parameters.items()
        ]
    }
    return TuningManifest.model_validate_json(json.dumps(fields))
