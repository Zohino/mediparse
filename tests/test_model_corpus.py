"""Otisk sdíleného modelového korpusu nezávisle na konkrétním nástroji."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.entrypoints import model_corpus

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "changed",
    ["config", "split", "candidates", "lexicon", "dataset_sha256", "packages"],
)
def test_checkpoint_key_changes_with_each_input(tmp_path: Path, changed: str) -> None:
    """Každý vědecký vstup a verze knihoven jsou součástí klíče."""
    inputs = {
        name: tmp_path / name for name in ("config", "split", "candidates", "lexicon")
    }
    for path in inputs.values():
        path.write_bytes(b"synthetic")
    corpus = model_corpus.ModelCorpusPaths(
        cohort=tmp_path / "cohort",
        split=inputs["split"],
        candidates=inputs["candidates"],
        lexicon=inputs["lexicon"],
    )
    packages = {"python": "test"}
    config = inputs["config"]
    before = model_corpus.checkpoint_key(config, corpus, "dataset", packages)
    assert model_corpus.checkpoint_key(config, corpus, "dataset", packages) == before
    dataset = "dataset"
    if changed == "dataset_sha256":
        dataset = "other"
    elif changed == "packages":
        packages = {"python": "other"}
    else:
        path = inputs[changed]
        path.write_bytes(path.read_bytes() + b" ")
    assert model_corpus.checkpoint_key(config, corpus, dataset, packages) != before
