"""Krok Snakemake: metriky z predikcí odložených zpráv."""

from pathlib import Path

from mediparse.entrypoints.model_evaluation import EvaluationPaths, main

paths = EvaluationPaths(
    predictions=Path(snakemake.input.predictions),
    cohort=Path(snakemake.input.cohort),
    split=Path(snakemake.input.split),
    training_manifest=Path(snakemake.input.training_manifest),
    metrics=Path(snakemake.output.metrics),
    manifest=Path(snakemake.output.manifest),
)

raise SystemExit(main(paths=paths, log=Path(snakemake.log[0])))
