"""Krok Snakemake: kohorta zpráv se štítky pětice a její manifest."""

from pathlib import Path

from mediparse.entrypoints.cohort import CohortPaths, main

paths = CohortPaths(
    discharge=Path(snakemake.input.discharge),
    diagnoses=Path(snakemake.input.diagnoses),
    candidates=Path(snakemake.input.candidates),
    corpus_rule=Path(snakemake.input.corpus_rule),
    cohort=Path(snakemake.output.cohort),
    manifest=Path(snakemake.output.manifest),
)

raise SystemExit(main(paths=paths, log=Path(snakemake.log[0])))
