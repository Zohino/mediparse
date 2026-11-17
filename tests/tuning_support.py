"""Syntetické konfigurace ladění se stejnými parametry jako produkční prostor."""

from __future__ import annotations

import json

from mediparse.domain.matrix import MatrixRow
from mediparse.domain.model_input import TextPreprocessing
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


def tuning_manifest_fields(**changes: object) -> dict[str, object]:
    """Sestaví společná syntetická pole manifestu pro validní i neplatné případy.

    Returns:
        Pole s rozpočtem tří pokusů, z nichž dva skončily a jeden byl prořezán.
    """
    training = training_manifest()
    return {
        "row_id": "row",
        "fold": 0,
        "dataset_sha256": "b" * 64,
        "config_sha256": "b" * 64,
        "matrix_sha256": "c" * 64,
        "split_sha256": "c" * 64,
        "split_seed": 23,
        "split_folds": 2,
        "seed": 7,
        "tuning_trials": 3,
        "completed_trials": 2,
        "pruned_trials": 1,
        "nonconverged_trials": 1,
        "best_macro_f1": 6 / 7,
        "best": SVM_PARAMS,
        "space": tuning_config().space.model_dump(),
        "sampler": "sampler",
        "pruner": "pruner",
        "seconds": 2.5,
        "train_notes": 40,
        "train_patients": 20,
        "diagnostic_strip": None,
        "text_preprocessing": TextPreprocessing.NONE,
        "text_normalization": None,
        "environment": training.environment.model_dump(),
        "hardware": training.hardware.model_dump(),
    } | changes


def tuning_manifest(
    *, best: dict[str, ParamValue] | None = None, **overrides: object
) -> TuningManifest:
    """Validuje společná pole manifestu s jednobodovým prostorem nejlepších parametrů.

    Returns:
        Manifest s přepsanými poli a zadanými nejlepšími parametry.
    """
    parameters = SVM_PARAMS if best is None else best
    fields = tuning_manifest_fields(
        **(
            {
                "best": parameters,
                "space": {
                    "params": [
                        {"kind": "categorical", "name": name, "choices": [value]}
                        for name, value in parameters.items()
                    ]
                },
            }
            | overrides
        )
    )
    return TuningManifest.model_validate_json(json.dumps(fields))
