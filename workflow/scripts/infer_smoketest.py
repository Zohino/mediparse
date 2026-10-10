"""Krok Snakemake: uložený model nad odloženými zprávami smoketestu."""

from pathlib import Path

from mediparse.entrypoints.smoketest_inference import main

raise SystemExit(
    main(
        notes=Path(snakemake.input.notes),
        model=Path(snakemake.input.model),
        predictions=Path(snakemake.input.predictions),
        demo=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
