"""Krok Snakemake: sdílený split pacientů zpráv kohorty."""

from pathlib import Path

from mediparse.entrypoints.split_materialization import main

raise SystemExit(
    main(
        cohort=Path(snakemake.input.cohort),
        config=Path(snakemake.input.config),
        output=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
