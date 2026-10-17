"""Krok Snakemake: sdílený split pacientů zpráv vstupní tabulky smoketestu."""

from pathlib import Path

from mediparse.entrypoints.split_materialization import main

raise SystemExit(
    main(
        notes=Path(snakemake.input.notes),
        config=Path(snakemake.input.config),
        output=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
