"""Krok Snakemake: model a predikce odložených zpráv ze vstupní tabulky smoketestu."""

from pathlib import Path

from mediparse.entrypoints.smoketest_training import main

raise SystemExit(
    main(
        notes=Path(snakemake.input.notes),
        config=Path(snakemake.input.config),
        model=Path(snakemake.output.model),
        predictions=Path(snakemake.output.predictions),
        manifest=Path(snakemake.output.manifest),
        log=Path(snakemake.log[0]),
    )
)
