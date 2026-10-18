"""Krok Snakemake: metriky smoketestu z predikcí odložených zpráv."""

from pathlib import Path

from mediparse.entrypoints.smoketest_evaluation import main

raise SystemExit(
    main(
        predictions=Path(snakemake.input.predictions),
        notes=Path(snakemake.input.notes),
        split=Path(snakemake.input.split),
        metrics=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
