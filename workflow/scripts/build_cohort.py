"""Krok Snakemake: kohorta zpráv se štítky pětice a její manifest."""

from pathlib import Path

from mediparse.entrypoints.cohort import main

raise SystemExit(
    main(
        discharge=Path(snakemake.input.discharge),
        diagnoses=Path(snakemake.input.diagnoses),
        candidates=Path(snakemake.input.candidates),
        tables=Path(snakemake.input.tables),
        cohort=Path(snakemake.output.cohort),
        manifest=Path(snakemake.output.manifest),
        log=Path(snakemake.log[0]),
    )
)
