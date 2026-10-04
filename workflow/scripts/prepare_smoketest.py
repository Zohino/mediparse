"""Krok Snakemake: vstupní tabulka smoketestu z plánů a anglických zpráv korpusu."""

from pathlib import Path

from mediparse.entrypoints.smoketest_input import main

raise SystemExit(
    main(
        plans=Path(snakemake.input.plans),
        corpus=Path(snakemake.params.corpus),
        output=Path(snakemake.output[0]),
        log=Path(snakemake.log[0]),
    )
)
