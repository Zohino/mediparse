"""Kontrakt screeningu: syntetická mřížka, predikce a jejich provenance pro R."""

from __future__ import annotations

import hashlib
from itertools import product
from random import Random
from typing import TYPE_CHECKING, Final

from mediparse.domain.classical_screening import ScreeningResult
from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification
from mediparse.domain.run_manifest import ScreeningManifest
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.screening_config import load_classical_screening_config
from mediparse.infrastructure.screening_table import (
    ParquetScreeningPredictions,
    ParquetScreeningTable,
)
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker
from tests.manifest_support import training_manifest
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path

CONTRACTS: Final = REPOSITORY / "tests/fixtures/contracts/screening"
CONFIG: Final = REPOSITORY / "config/screening_classical.json"
THRESHOLD: Final = 0.5


def screening_cells() -> tuple[ScreeningResult, ...]:
    """Sestaví úplnou mřížku configu nad 24 zprávami v každém foldu.

    Prefix eda-sentinel je svázaný s prefix=eda-sentinel v eda-synthetic.sh,
    aby kanárek odhalil únik identit i do freeze a exportů.

    Returns:
        Deterministické buňky s oběma třídami a rozdílnou chybovostí rodin.
    """
    config = load_classical_screening_config(CONFIG)
    cells = []
    grid = product(range(config.folds), config.vectorizers, config.families, Diagnosis)
    for index, (fold, vectorizer, family, diagnosis) in enumerate(grid):
        random = Random(index)
        error_rate = 0.08 + 0.05 * config.families.index(family)
        labels = tuple(n % 2 == 0 for n in range(24))
        scores = tuple(
            random.uniform(0.55, 0.95)
            if label != (random.random() < error_rate)
            else random.uniform(0.05, 0.45)
            for label in labels
        )
        classifications = tuple(
            Classification(score >= THRESHOLD, score) for score in scores
        )
        cells.append(
            ScreeningResult(
                fold=fold,
                vectorizer=vectorizer.name,
                family=family,
                diagnosis=diagnosis,
                note_ids=tuple(f"eda-sentinel-{fold}-{n}" for n in range(24)),
                labels=labels,
                classifications=classifications,
                threshold=THRESHOLD,
                converged=index != 0,
                metrics=SklearnScorer.score(
                    labels, tuple(item.predicted for item in classifications), scores
                ),
            )
        )
    return tuple(cells)


def write_screening_fixtures(directory: Path) -> None:
    """Zapíše kontraktové vstupy stejnými zapisovači jako skutečný screening.

    Přegenerování přes pixi run --frozen python -c:
    'from tests.test_screening_contract_fixtures import CONTRACTS, write_screening_fixtures;
    write_screening_fixtures(CONTRACTS)'.
    """
    config = load_classical_screening_config(CONFIG)
    cells = screening_cells()
    results = directory / "classical.parquet"
    predictions = directory / "classical_predictions.parquet"
    ParquetScreeningTable(results).write(cells)
    ParquetScreeningPredictions(predictions).write(cells)
    manifest = ScreeningManifest(
        dataset_sha256="a" * 64,
        split_seed=0,
        split_folds=5,
        split_sha256="b" * 64,
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        seed=config.seed,
        results_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
        predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        development_notes=24 * config.folds,
        development_patients=24 * config.folds,
        diagnostic_strip=None,
        text_preprocessing=config.input.text_preprocessing,
        text_normalization=TextNormalization(acronyms={}, variants={}),
        environment=training_manifest().environment,
    )
    write_model(directory / "classical_manifest.json", manifest)
    write_marker(
        directory,
        SyntheticMarker(
            prefix="eda-sentinel", rows=sum(len(c.note_ids) for c in cells)
        ),
    )


def test_screening_fixtures_match_writers_and_lineage(tmp_path: Path) -> None:
    """Fixture odpovídají producentům a manifest skutečným SHA-256 vstupů."""
    write_screening_fixtures(tmp_path)
    for name in (
        "classical.parquet",
        "classical_predictions.parquet",
        "classical_manifest.json",
        "synthetic.json",
    ):
        assert (tmp_path / name).read_bytes() == (CONTRACTS / name).read_bytes()
    manifest = ScreeningManifest.model_validate_json(
        (CONTRACTS / "classical_manifest.json").read_bytes()
    )
    for field, path in (
        ("results_sha256", CONTRACTS / "classical.parquet"),
        ("predictions_sha256", CONTRACTS / "classical_predictions.parquet"),
        ("config_sha256", CONFIG),
    ):
        assert getattr(manifest, field) == hashlib.sha256(path.read_bytes()).hexdigest()
