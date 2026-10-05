"""Krok Snakemake: model a held-out metriky jedné diagnózy ze vstupní tabulky smoketestu."""

from pathlib import Path

from mediparse.entrypoints.smoketest_training import main

raise SystemExit(
    main(
        notes=Path(snakemake.input.notes),
        config=Path(snakemake.input.config),
        model=Path(snakemake.output.model),
        metrics=Path(snakemake.output.metrics),
        log=Path(snakemake.log[0]),
    )
)
