"""Krok Snakemake: model a predikce odložených zpráv z kohorty."""

from pathlib import Path

from mediparse.entrypoints.model_training import main

tuning = snakemake.input.get("tuning")

raise SystemExit(
    main(
        cohort=Path(snakemake.input.cohort),
        config=Path(snakemake.input.config),
        candidates=Path(snakemake.input.candidates),
        lexicon=Path(snakemake.input.lexicon),
        split=Path(snakemake.input.split),
        tuning=Path(tuning) if tuning is not None else None,
        model=Path(snakemake.output.model),
        predictions=Path(snakemake.output.predictions),
        manifest=Path(snakemake.output.manifest),
        costs=Path(snakemake.output.costs),
        log=Path(snakemake.log[0]),
    )
)
