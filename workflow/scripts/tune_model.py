"""Krok Snakemake: ladění jednoho řádku matice uvnitř tréninku foldu."""

from pathlib import Path

from mediparse.entrypoints.model_tuning import main

raise SystemExit(
    main(
        cohort=Path(snakemake.input.cohort),
        config=Path(snakemake.input.config),
        matrix=Path(snakemake.input.matrix),
        candidates=Path(snakemake.input.candidates),
        lexicon=Path(snakemake.input.lexicon),
        split=Path(snakemake.input.split),
        storage=Path(snakemake.output.storage),
        manifest=Path(snakemake.output.manifest),
        log=Path(snakemake.log[0]),
    )
)
