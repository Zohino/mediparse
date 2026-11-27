"""Argumenty a otisk sdíleného modelového korpusu nezávisle na nástroji."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mediparse.entrypoints import model_corpus


@pytest.mark.parametrize("missing", ["cohort", "split", "candidates", "lexicon"])
def test_corpus_arguments_are_required(
    missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Parser odmítne vynechání kterékoli cesty korpusu."""
    parser = argparse.ArgumentParser()
    model_corpus.ModelCorpusPaths.add_arguments(parser)
    argv = []
    for name in ("cohort", "split", "candidates", "lexicon"):
        if name != missing:
            argv.extend((f"--{name}", name))
    with pytest.raises(SystemExit):
        parser.parse_args(argv)
    assert f"--{missing}" in capsys.readouterr().err


def test_corpus_paths_from_arguments() -> None:
    """Úplné argumenty sestaví všechny cesty se správným přiřazením a typem."""
    parser = argparse.ArgumentParser()
    model_corpus.ModelCorpusPaths.add_arguments(parser)
    args = parser.parse_args([
        "--cohort",
        "c",
        "--split",
        "s",
        "--candidates",
        "k",
        "--lexicon",
        "l",
    ])
    assert model_corpus.ModelCorpusPaths.from_arguments(args) == (
        model_corpus.ModelCorpusPaths(
            cohort=Path("c"), split=Path("s"), candidates=Path("k"), lexicon=Path("l")
        )
    )


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
