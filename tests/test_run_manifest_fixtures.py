"""Kontrakt JSON manifestů: pevná syntetická provenance zapsaná producentem Pythonu."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.run_manifest import EvaluationManifest, RunManifest
from mediparse.infrastructure.json_file import write_model
from tests.manifest_support import training_manifest
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path

CONTRACTS: Final = REPOSITORY / "tests" / "fixtures" / "contracts"


def evaluation_manifest() -> EvaluationManifest:
    """Sestaví pevný manifest evaluace navazující na bajty fixture tréninku a metrik.

    Returns:
        Manifest s otisky vypočtenými nezávislým orákulem hashlib.
    """
    training = training_manifest(row_id="smoketest")
    return EvaluationManifest(
        row_id=training.row_id,
        training_manifest_sha256=hashlib.sha256(
            (CONTRACTS / "run_manifest.json").read_bytes()
        ).hexdigest(),
        metrics_sha256=hashlib.sha256(
            (CONTRACTS / "metrics.parquet").read_bytes()
        ).hexdigest(),
        predictions_sha256=training.predictions_sha256,
        dataset_sha256=training.dataset_sha256,
        split_sha256=training.split_sha256,
        environment=training.environment,
    )


@pytest.mark.parametrize("name", ["run_manifest.json", "evaluation_manifest.json"])
def test_shared_manifest_fixture_matches_writer_and_lineage(
    tmp_path: Path, name: str
) -> None:
    """Fixture projde přísným modelem, odpovídá zapisovači a váže správné vstupy.

    Přegenerování: write_model(CONTRACTS / "run_manifest.json", training_manifest(row_id="smoketest"))
    a potom write_model(CONTRACTS / "evaluation_manifest.json", evaluation_manifest()).
    """
    expected = (
        training_manifest(row_id="smoketest")
        if name == "run_manifest.json"
        else evaluation_manifest()
    )
    model = RunManifest if name == "run_manifest.json" else EvaluationManifest
    fixture = (CONTRACTS / name).read_bytes()
    assert model.model_validate_json(fixture) == expected
    output = tmp_path / name
    write_model(output, expected)
    assert output.read_bytes() == fixture
