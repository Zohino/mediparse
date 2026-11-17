"""Krok Snakemake: ladění jednoho řádku matice uvnitř tréninku foldu."""

from pathlib import Path

from mediparse.entrypoints.model_input_table import ModelCorpusPaths
from mediparse.entrypoints.model_tuning import TuningPaths, main

paths = TuningPaths(
    corpus=ModelCorpusPaths(
        cohort=Path(snakemake.input.cohort),
        candidates=Path(snakemake.input.candidates),
        lexicon=Path(snakemake.input.lexicon),
        split=Path(snakemake.input.split),
    ),
    config=Path(snakemake.input.config),
    matrix=Path(snakemake.input.matrix),
    storage=Path(snakemake.output.storage),
    manifest=Path(snakemake.output.manifest),
)

raise SystemExit(main(paths=paths, log=Path(snakemake.log[0])))
