"""Krok Snakemake: validovaná tabulka MIMIC z CSV na parquet."""

from pathlib import Path

from mediparse.entrypoints.mimic_parquet import main

raise SystemExit(
    main(
        table=Path(snakemake.input.table),
        inventory=Path(snakemake.input.inventory),
        parquet=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
