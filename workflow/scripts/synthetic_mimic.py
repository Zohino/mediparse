"""Krok Snakemake: syntetické tabulky MIMIC a otisky vstupů kohorty smoketestu."""

from pathlib import Path

from mediparse.synthetic.entrypoints.synthetic_mimic import SyntheticMimicPaths, main

paths = SyntheticMimicPaths(
    plans=Path(snakemake.input.plans),
    corpus=Path(snakemake.params.corpus),
    candidates=Path(snakemake.input.candidates),
    manifest=Path(snakemake.params.manifest),
    discharge=Path(snakemake.output.discharge),
    diagnoses=Path(snakemake.output.diagnoses),
    artifacts=Path(snakemake.output.artifacts_marker).parent,
)

raise SystemExit(main(paths=paths, log=Path(snakemake.log[0])))
