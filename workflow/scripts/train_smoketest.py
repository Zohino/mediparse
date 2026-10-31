"""Krok Snakemake: model a predikce odložených zpráv ze vstupní tabulky smoketestu."""

from pathlib import Path

from mediparse.entrypoints.model_training import main

raise SystemExit(
    main(
        cohort=Path(snakemake.input.notes),
        config=Path(snakemake.input.config),
        candidates=Path(snakemake.input.candidates),
        lexicon=Path(snakemake.input.lexicon),
        split=Path(snakemake.input.split),
        model=Path(snakemake.output.model),
        predictions=Path(snakemake.output.predictions),
        manifest=Path(snakemake.output.manifest),
        log=Path(snakemake.log[0]),
    )
)
