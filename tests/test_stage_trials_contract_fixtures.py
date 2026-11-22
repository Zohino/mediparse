"""Syntetický kontrakt pokusů stupně a jeho provenance pro R."""

from __future__ import annotations

import hashlib
from random import Random
from typing import TYPE_CHECKING, Final

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.run_manifest import StageTuningManifest
from mediparse.domain.stage_tuning import StageFoldResult, StageTrialRecord, best_trial
from mediparse.domain.tuning import macro_max_f1, max_f1
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.stage_config import load_stage_config
from mediparse.infrastructure.stage_trials_table import ParquetStageCheckpoint
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker
from tests.manifest_support import training_manifest
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path

CONTRACTS: Final = REPOSITORY / "tests/fixtures/contracts/stage_trials"
CONFIG: Final = REPOSITORY / "config/stage_classical.json"


def test_stage_trials_fixtures_match_writers_and_manifest_model(tmp_path: Path) -> None:
    """Fixture odpovídají zapisovači a manifest projde doménovým modelem."""
    write_stage_trials_fixtures(tmp_path)
    for name in (
        "classical_trials.parquet",
        "classical_trial_predictions.parquet",
        "classical_trials_manifest.json",
        "synthetic.json",
    ):
        assert (tmp_path / name).read_bytes() == (CONTRACTS / name).read_bytes()
    StageTuningManifest.model_validate_json(
        (CONTRACTS / "classical_trials_manifest.json").read_bytes()
    )


def write_stage_trials_fixtures(directory: Path) -> None:
    """Zapíše kontrakt stejným zapisovačem jako ladění stupně.

    Přegenerování přes pixi run --frozen python -c:
    'from tests.test_stage_trials_contract_fixtures import CONTRACTS,
    write_stage_trials_fixtures; write_stage_trials_fixtures(CONTRACTS)'.
    Prefix eda-sentinel váže identity na kanárek v eda-synthetic.sh.
    """
    config = load_stage_config(CONFIG)
    results = directory / "classical_trials.parquet"
    predictions = directory / "classical_trial_predictions.parquet"
    checkpoint = ParquetStageCheckpoint(
        directory / "checkpoint", "synthetic", results, predictions
    )
    records = []
    rows = 0
    for trial in range(config.trials):
        branch = config.branches[trial % len(config.branches)]
        family_index = config.families.index(branch.family)
        error_rate = (
            0.08 + 0.05 * family_index - 0.002 * (trial // len(config.branches))
        )
        cells, value = stage_cells(
            trial,
            error_rate,
            config.folds,
            config.lsa.max_features if config.lsa else 1000,
        )
        rows += sum(len(cell.note_ids) for cell in cells)
        record = StageTrialRecord(
            number=trial,
            params=branch.default_params(),
            default=trial < len(config.branches),
            value=value,
            converged=trial != 0,
            seconds=float(trial + 1),
        )
        records.append(record)
        checkpoint.write(record, cells)
    checkpoint.finish()
    winner = best_trial(records)
    provenance = training_manifest()
    manifest = StageTuningManifest(
        jobs=1,
        dataset_sha256="a" * 64,
        split_seed=0,
        split_folds=5,
        split_sha256="b" * 64,
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        seed=config.seed,
        trials=config.trials,
        random_startup_trials=config.random_startup_trials,
        completed_trials=len(records),
        nonconverged_trials=1,
        best_trial=winner.number,
        best_branch=str(winner.params["branch"]),
        best_params=dict(winner.params),
        best_macro_f1=winner.value,
        sampler="TPESampler(multivariate=True, group=True)",
        pruner="NopPruner",
        seconds=sum(r.seconds for r in records),
        sample_notes=24 * config.folds,
        sample_patients=24 * config.folds,
        development_notes=24 * config.folds * config.subsample_parts,
        development_patients=24 * config.folds * config.subsample_parts,
        results_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
        predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        diagnostic_strip=None,
        text_preprocessing=config.text_preprocessing,
        text_normalization=dict.fromkeys(
            config.text_preprocessing, TextNormalization(acronyms={}, variants={})
        ),
        environment=provenance.environment,
        hardware=provenance.hardware,
    )
    write_model(directory / "classical_trials_manifest.json", manifest)
    write_marker(directory, SyntheticMarker(prefix="eda-sentinel", rows=rows))


def stage_cells(
    trial: int, error_rate: float, folds: int, cap: int
) -> tuple[list[StageFoldResult], float]:
    """Sestaví buňky pokusu a cíl jako průměr foldových macro-MaxF1.

    Returns:
        Buňky se syntetickými identitami a cíl pokusu.
    """
    cells = []
    fold_values = []
    for fold in range(folds):
        random = Random(trial * folds + fold)
        labels = tuple(n % 2 == 0 for n in range(24))
        metrics = {}
        for diagnosis in Diagnosis:
            scores = tuple(
                random.uniform(0.55, 0.95)
                if label != (random.random() < error_rate)
                else random.uniform(0.05, 0.45)
                for label in labels
            )
            metrics[diagnosis] = max_f1(labels, scores)
            cells.append(
                StageFoldResult(
                    fold=fold,
                    diagnosis=diagnosis,
                    note_ids=tuple(f"eda-sentinel-{fold}-{n}" for n in range(24)),
                    labels=labels,
                    scores=scores,
                    max_f1=metrics[diagnosis],
                    converged=trial != 0,
                    n_features=cap if fold == 0 else 1000,
                )
            )
        fold_values.append(macro_max_f1(metrics))
    return cells, sum(fold_values) / len(fold_values)
