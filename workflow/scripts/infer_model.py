"""Krok Snakemake: uložený model nad odloženými zprávami."""

from pathlib import Path

from mediparse.entrypoints.model_inference import main

raise SystemExit(
    main(
        cohort=Path(snakemake.input.cohort),
        config=Path(snakemake.input.config),
        lexicon=Path(snakemake.input.lexicon),
        model=Path(snakemake.input.model),
        predictions=Path(snakemake.input.predictions),
        inference=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
