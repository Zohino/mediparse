"""Syntetický průchod pilotem, přístupová stráž a manifest."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest

from mediparse.domain.run_manifest import TransformerPilotManifest
from mediparse.entrypoints import transformer_pilot
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.split_table import ParquetSplitTable
from mediparse.infrastructure.synthetic_marker import MARKER_NAME
from mediparse.infrastructure.transformer_pilot_table import PILOT_SCHEMA
from tests.pilot_support import Source, Trainer, corpus, pilot_config
from tests.stage_support import StepClock
from tests.support import fixed_revision, mark_synthetic, repository_corpus

if TYPE_CHECKING:
    from pathlib import Path


def _paths(
    root: Path, *, strip: bool = False
) -> transformer_pilot.TransformerPilotPaths:
    cohort, split, config = (
        root / name for name in ("cohort.parquet", "split.parquet", "config.json")
    )
    ParquetNoteTable(cohort).write(corpus())
    ParquetSplitTable(split).write(Source().read())
    config.write_text(
        pilot_config()
        .model_copy(update={"strip_diagnostic_sections": strip})
        .model_dump_json(),
        encoding="utf-8",
    )
    mark_synthetic(root)
    return transformer_pilot.TransformerPilotPaths(
        corpus=repository_corpus(cohort, split),
        config=config,
        out=root / "out/pilot.parquet",
        manifest=root / "out/manifest.json",
    )


def _run(paths: transformer_pilot.TransformerPilotPaths) -> ExitCode:
    return transformer_pilot.run(
        environ={"CODEX_THREAD_ID": "test"},
        paths=paths,
        revision=fixed_revision,
        clock=StepClock(),
        trainer_factory=lambda _seed: Trainer(),
    )


@pytest.mark.parametrize("strip", [False, True])
def test_end_to_end_manifest_preserves_diagnostic_strip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, strip: bool
) -> None:
    """Parquet obsahuje jen agregáty a manifest skutečné otisky i volbu stripu."""
    paths = _paths(tmp_path, strip=strip)
    monkeypatch.setattr(transformer_pilot, "require_device", lambda _device: None)
    monkeypatch.setattr(
        transformer_pilot,
        "installed_versions",
        lambda packages: dict.fromkeys(packages, "test"),
    )
    loader = Mock(wraps=transformer_pilot.model_corpus)
    monkeypatch.setattr(transformer_pilot, "model_corpus", loader)
    assert _run(paths) is ExitCode.OK
    assert loader.call_args.kwargs["strip_diagnostic_sections"] is strip
    table = pq.read_table(paths.out)
    assert table.schema == PILOT_SCHEMA
    assert pq.read_metadata(paths.out).row_group(0).column(0).compression == "ZSTD"
    assert table.num_rows == 6
    assert all(row["macro_f1"] == pytest.approx(0.8) for row in table.to_pylist())
    manifest = TransformerPilotManifest.model_validate_json(paths.manifest.read_bytes())
    for field, path in (
        ("results_sha256", paths.out),
        ("config_sha256", paths.config),
        ("dataset_sha256", paths.corpus.cohort),
    ):
        assert getattr(manifest, field) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "runs" not in manifest.model_dump()
    assert manifest.encoders == tuple(run.encoder for run in pilot_config().runs)
    assert manifest.split_sha256 == Source().read().split.sha256()
    assert manifest.environment.source == fixed_revision()
    assert set(manifest.environment.packages) >= {
        "torch",
        "transformers",
        "tokenizers",
        "python",
    }
    if strip:
        assert manifest.diagnostic_strip is not None
        assert (
            manifest.diagnostic_strip.lexicon_sha256
            == hashlib.sha256(paths.corpus.lexicon.read_bytes()).hexdigest()
        )
    else:
        assert manifest.diagnostic_strip is None


def test_access_is_refused_before_reading(tmp_path: Path) -> None:
    """Blokující prostředí odmítne neoznačený korpus bez výstupů."""
    paths = _paths(tmp_path)
    (tmp_path / MARKER_NAME).unlink()
    assert _run(paths) is ExitCode.REFUSED
    assert not paths.out.exists()
    assert not paths.manifest.exists()


def test_missing_torch_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chybějící torch se odmítne před čtením kohorty a před zjištěním verzí."""
    paths = _paths(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    paths = replace(
        paths, corpus=replace(paths.corpus, cohort=tmp_path / "missing.parquet")
    )
    packages, reader = Mock(), Mock()
    monkeypatch.setattr(transformer_pilot, "installed_versions", packages)
    monkeypatch.setattr(transformer_pilot, "model_corpus", reader)
    assert _run(paths) is ExitCode.REFUSED
    packages.assert_not_called()
    reader.assert_not_called()
    assert not paths.out.exists()
