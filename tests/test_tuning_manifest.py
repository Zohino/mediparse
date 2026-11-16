"""Manifest ladění hlídá rozpočet, prostor i soulad předzpracování."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.environment import Environment, Hardware
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import TuningManifest
from tests.support import REVISION
from tests.tuning_support import SVM_PARAMS, tuning_config


def _fields(**changes: object) -> dict[str, object]:
    return {
        "row_id": "en-svm",
        "fold": 0,
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "matrix_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "split_seed": 23,
        "split_folds": 2,
        "seed": 7,
        "tuning_trials": 3,
        "completed_trials": 2,
        "pruned_trials": 1,
        "nonconverged_trials": 1,
        "best_macro_f1": 6 / 7,
        "best": SVM_PARAMS,
        "space": tuning_config().space,
        "sampler": "sampler",
        "pruner": "pruner",
        "seconds": 2.5,
        "train_notes": 40,
        "train_patients": 20,
        "diagnostic_strip": None,
        "text_preprocessing": TextPreprocessing.NONE,
        "text_normalization": None,
        "environment": Environment(packages={"python": "3.13.0"}, source=REVISION),
        "hardware": Hardware(
            cpu_model="synthetic",
            logical_cores=2,
            available_cores=2,
            memory_bytes=1024,
            gpus=(),
        ),
    } | changes


def test_tuning_manifest_roundtrips() -> None:
    """Povinné agregáty a provenance přežijí JSON beze změny."""
    manifest = TuningManifest.model_validate(_fields())
    assert TuningManifest.model_validate_json(manifest.model_dump_json()) == manifest


@pytest.mark.parametrize(
    "changes",
    [
        {"pruned_trials": 0},
        {"best": SVM_PARAMS | {"c": 1000.0}},
        {"nonconverged_trials": 4},
        {"completed_trials": -1, "pruned_trials": 4},
        {"best_macro_f1": 1.1},
        {"seconds": -1.0},
        {"text_preprocessing": TextPreprocessing.DICTIONARY},
        {"text_normalization": TextNormalization(acronyms={}, variants={})},
        {
            "text_preprocessing": TextPreprocessing.NORMALIZED,
            "text_normalization": TextNormalization(
                acronyms={}, variants={"kindey": "kidney"}
            ),
        },
    ],
)
def test_tuning_manifest_rejects_inconsistent_record(
    changes: dict[str, object],
) -> None:
    """Rozpočet 3 odmítne 2+0, nejlepší hodnoty mimo prostor i rozpornou normalizaci."""
    with pytest.raises(ValidationError):
        TuningManifest.model_validate(_fields(**changes))
