"""Pevná syntetická provenance pro testy manifestů a jejich kontraktu s R."""

from mediparse.domain.environment import Environment, Hardware
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import (
    RefinementCandidate,
    RunManifest,
    StageRefinementManifest,
)
from tests.model_support import FIXED_SVM_PARAMS
from tests.support import REVISION


def training_manifest(
    *,
    row_id: str = "row",
    predictions_sha256: str = "a" * 64,
    dataset_sha256: str = "b" * 64,
    split_sha256: str = "c" * 64,
) -> RunManifest:
    """Sestaví úplný manifest z pevných hodnot a dodaných otisků.

    Returns:
        Syntetický manifest bez závislosti na stroji a instalovaných verzích.
    """
    return RunManifest(
        row_id=row_id,
        predictions_sha256=predictions_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        config_sha256="d" * 64,
        parameters=FIXED_SVM_PARAMS,
        threshold=0.25,
        threshold_folds=3,
        converged=True,
        seed=0,
        split_seed=0,
        split_folds=2,
        diagnostic_strip=None,
        text_preprocessing=TextPreprocessing.NONE,
        text_normalization=None,
        model="synthetic classifier",
        model_revision=None,
        tokenizer_revision=None,
        environment=Environment(packages={"python": "3.13.0"}, source=REVISION),
        hardware=Hardware(
            cpu_model="Synthetic CPU",
            logical_cores=8,
            available_cores=2,
            memory_bytes=1024,
            gpus=(),
        ),
    )


def refinement_manifest(**changes: object) -> StageRefinementManifest:
    """Sestaví manifest dvou syntetických rodin s různým cílem kontroly.

    Returns:
        Manifest s vítěznou druhou větví a pevným prostředím.
    """
    training = training_manifest()
    candidates = tuple(
        RefinementCandidate(
            stage_trial=i,
            branch=branch,
            params={"branch": branch},
            stage_macro_f1=0.9 - i / 10,
            check_macro_f1=value,
        )
        for i, (branch, value) in enumerate((
            ("word/logreg", 0.6),
            ("word/linear_svm", 0.8),
        ))
    )
    return StageRefinementManifest.model_validate(
        {
            "dataset_sha256": "a" * 64,
            "stage_config_sha256": "b" * 64,
            "refinement_config_sha256": "c" * 64,
            "matrix_sha256": "d" * 64,
            "stage_manifest_sha256": "e" * 64,
            "stage_results_sha256": "f" * 64,
            "split_seed": 0,
            "split_folds": 2,
            "split_sha256": "a" * 64,
            "seed": 0,
            "folds": 2,
            "jobs": 1,
            "tuning_trials": 6,
            "random_startup_trials": 1,
            "candidates": candidates,
            "best_trial": 1,
            "best_params": {"branch": "word/linear_svm"},
            "best_macro_f1": 0.8,
            "completed_trials": 7,
            "nonconverged_trials": 0,
            "sampler": "synthetic sampler",
            "pruner": "no pruning",
            "seconds": 1.0,
            "development_notes": 100,
            "development_patients": 100,
            "results_sha256": "b" * 64,
            "predictions_sha256": "c" * 64,
            "diagnostic_strip": None,
            "text_preprocessing": {"word": TextPreprocessing.NONE},
            "text_normalization": {"word": None},
            "environment": training.environment,
            "hardware": training.hardware,
        }
        | changes
    )
