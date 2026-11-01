"""Krok Snakemake: souhrn metrik a ukázky predikce smoketestu na stdout."""

from pathlib import Path

from mediparse.entrypoints.smoketest_summary import main

raise SystemExit(
    main(
        metrics=Path(snakemake.input.metrics),
        inference=Path(snakemake.input.inference),
        corpus=Path(snakemake.params.corpus),
        outputs=Path(snakemake.params.outputs),
        logs=Path(snakemake.params.logs),
    )
)
