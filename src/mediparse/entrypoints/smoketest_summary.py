"""Vstupní bod pravidla Snakemake ``smoketest``: souhrn metrik a ukázky predikce na výstup.

Cesty k metrikám a ukázce jsou vstupy pravidla; skript workflow jen rozbalí objekt
``snakemake`` a zavolá ``main``. Výpis jde na stdout, aby ho zkoušející viděl
na konci běhu.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, TextIO

from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.metrics_table import ParquetMetricsTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable

if TYPE_CHECKING:
    from pathlib import Path

HEADER: Final = ("řádek matice", "diagnóza", "zprávy v testu", "pozitivní")
SCORES: Final = ("F1", "ROC-AUC", "PR-AUC")
DEMO_HEADER: Final = ("zpráva", "diagnóza", "pravda", "predikce", "skóre")
CORPUS: Final = "resources/synthetic/en"
YES_NO: Final = {True: "ano", False: "ne"}
OUTPUTS: Final = ("build/smoketest/", "logs/smoketest/")


def main(*, metrics: Path, demo: Path) -> ExitCode:
    """Krok pravidla: souhrn na stdout, diagnostika na stderr.

    Returns:
        Návratový kód kroku.
    """
    configure_logging()
    return run(metrics=metrics, demo=demo, out=sys.stdout)


def _write_table(out: TextIO, table: list[tuple[str, ...]]) -> None:
    widths = [max(len(row[column]) for row in table) for column in range(len(table[0]))]
    for row in table:
        cells = zip(row, widths, strict=True)
        out.write("  ".join(cell.ljust(width) for cell, width in cells).rstrip() + "\n")


@refusing_invalid_input
def run(*, metrics: Path, demo: Path, out: TextIO) -> ExitCode:
    """Přečte metriky a ukázku predikce a vypíše souhrn s cestami k výstupům.

    Returns:
        OK po výpisu; REFUSED pro chybějící nebo neplatnou tabulku metrik či ukázky.
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
    examples = [
        (
            f"{CORPUS}/{item.note_id}.txt",
            item.diagnosis.value,
            YES_NO[item.truth],
            YES_NO[item.predicted],
            f"{item.score:.3f}",
        )
        for item in ParquetPredictionTable(demo).read()
    ]
    _write_table(out, [(*HEADER, *SCORES), *rows])
    out.write("\nUkázka predikce uloženého modelu na odložených zprávách")
    out.write(" (kladné skóre znamená ano):\n")
    _write_table(out, [DEMO_HEADER, *examples])
    out.write(f"\nData, model, predikce, metriky a manifest: {OUTPUTS[0]}\n")
    out.write(f"Logy kroků: {OUTPUTS[1]}\n")
    return ExitCode.OK
