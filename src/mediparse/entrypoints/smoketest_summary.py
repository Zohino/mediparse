"""Vstupní bod pravidla Snakemake ``smoketest``: souhrn metrik smoketestu na výstup.

Cesta k metrikám je vstup pravidla; skript workflow jen rozbalí objekt
``snakemake`` a zavolá ``main``. Výpis jde na stdout, aby ho zkoušející viděl
na konci běhu.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, TextIO

from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.metrics_table import ParquetMetricsTable

if TYPE_CHECKING:
    from pathlib import Path

HEADER: Final = ("řádek matice", "diagnóza", "zprávy v testu", "pozitivní")
SCORES: Final = ("F1", "ROC-AUC", "PR-AUC")
OUTPUTS: Final = ("build/smoketest/", "logs/smoketest/")


def main(*, metrics: Path) -> ExitCode:
    """Krok pravidla: souhrn na stdout, diagnostika na stderr.

    Returns:
        Návratový kód kroku.
    """
    configure_logging()
    return run(metrics=metrics, out=sys.stdout)


@refusing_invalid_input
def run(*, metrics: Path, out: TextIO) -> ExitCode:
    """Přečte tabulku metrik a vypíše z ní souhrn s cestami k výstupům.

    Returns:
        OK po výpisu; REFUSED pro chybějící nebo neplatnou tabulku metrik.
    """
    rows = [
        (
            item.row_id,
            item.diagnosis.value,
            str(item.test_notes),
            str(item.test_positives),
            f"{item.metrics.f1:.3f}",
            f"{item.metrics.roc_auc:.3f}",
            f"{item.metrics.pr_auc:.3f}",
        )
        for item in ParquetMetricsTable(metrics).read()
    ]
    table = [(*HEADER, *SCORES), *rows]
    widths = [max(len(row[column]) for row in table) for column in range(len(table[0]))]
    for row in table:
        cells = zip(row, widths, strict=True)
        out.write("  ".join(cell.ljust(width) for cell, width in cells).rstrip() + "\n")
    out.write(f"\nData, model, predikce, metriky a manifest: {OUTPUTS[0]}\n")
    out.write(f"Logy kroků: {OUTPUTS[1]}\n")
    return ExitCode.OK
