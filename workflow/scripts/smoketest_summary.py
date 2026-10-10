"""Krok Snakemake: souhrn metrik smoketestu na stdout."""

from pathlib import Path

from mediparse.entrypoints.smoketest_summary import main

raise SystemExit(main(metrics=Path(snakemake.input.metrics)))
