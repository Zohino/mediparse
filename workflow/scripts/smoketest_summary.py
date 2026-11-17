"""Krok Snakemake: souhrn metrik a ukázky predikce smoketestu na stdout."""

from pathlib import Path

from mediparse.synthetic.entrypoints.smoketest_summary import SmoketestPaths, main

paths = SmoketestPaths(
    metrics=Path(snakemake.input.metrics),
    inference=Path(snakemake.input.inference),
    cohort=Path(snakemake.input.cohort),
    corpus=Path(snakemake.params.corpus),
    outputs=Path(snakemake.params.outputs),
    logs=Path(snakemake.params.logs),
)

raise SystemExit(main(paths=paths))
