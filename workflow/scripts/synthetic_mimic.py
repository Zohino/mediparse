"""Krok Snakemake: syntetické tabulky MIMIC a otisky vstupů kohorty smoketestu."""

from pathlib import Path

from mediparse.entrypoints.synthetic_mimic import main

raise SystemExit(
    main(
        plans=Path(snakemake.input.plans),
        corpus=Path(snakemake.params.corpus),
        candidates=Path(snakemake.input.candidates),
        manifest=Path(snakemake.params.manifest),
        discharge=Path(snakemake.output.discharge),
        diagnoses=Path(snakemake.output.diagnoses),
        tables=Path(snakemake.output.tables),
        log=Path(snakemake.log[0]),
    )
)
