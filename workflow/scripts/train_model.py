"""Krok Snakemake: model a predikce odložených zpráv z kohorty."""

from pathlib import Path

from mediparse.entrypoints.model_corpus import ModelCorpusPaths
from mediparse.entrypoints.model_training import TrainingPaths, main

paths = TrainingPaths(
    corpus=ModelCorpusPaths(
        cohort=Path(snakemake.input.cohort),
        candidates=Path(snakemake.input.candidates),
        lexicon=Path(snakemake.input.lexicon),
        split=Path(snakemake.input.split),
    ),
    config=Path(snakemake.input.config),
    model=Path(snakemake.output.model),
    predictions=Path(snakemake.output.predictions),
    manifest=Path(snakemake.output.manifest),
    costs=Path(snakemake.output.costs),
)

raise SystemExit(main(paths=paths, log=Path(snakemake.log[0])))
