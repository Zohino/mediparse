"""Vstupní bod syntetických tabulek MIMIC pro kohortu smoketestu."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.domain.language import Language
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.table_provenance import write_table_provenance
from mediparse.synthetic.domain.synthetic_mimic import input_notes, synthetic_mimic_rows
from mediparse.synthetic.infrastructure.plans_file import PlansFile
from mediparse.synthetic.infrastructure.synthetic_corpus import CorpusDirectory

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    *,
    plans: Path,
    corpus: Path,
    candidates: Path,
    manifest: Path,
    discharge: Path,
    diagnoses: Path,
    log: Path,
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(
        plans=plans,
        corpus=corpus,
        candidates=candidates,
        manifest=manifest,
        discharge=discharge,
        diagnoses=diagnoses,
    )


@refusing_invalid_input
def run(
    *,
    plans: Path,
    corpus: Path,
    candidates: Path,
    manifest: Path,
    discharge: Path,
    diagnoses: Path,
) -> ExitCode:
    """Zapíše tabulky discharge a diagnoses_icd a jejich otisky pro manifest kohorty.

    Returns:
        OK po zápisu; REFUSED pro neplatné vstupy nebo nespárované plány a zprávy.
    """
    notes = input_notes(
        PlansFile(plans).load(), CorpusDirectory(corpus).notes(Language.EN), Language.EN
    )
    shapes = read_manifest(manifest)
    rows = synthetic_mimic_rows(notes, load_diagnosis_codes(candidates), shapes)
    outputs = {"discharge": discharge, "diagnoses_icd": diagnoses}
    for table, path in outputs.items():
        write_string_table(path, shapes[table].columns, rows[table])
        # Syntetika nemá CSV; .invalid odliší její původ od skutečných tabulek MIMIC.
        write_table_provenance(
            path,
            source=f"https://mediparse.invalid/synthetic/{table}",
            source_sha256=file_sha256(path),
        )
    logger.info(
        "Syntetické tabulky MIMIC: %d zpráv v %s.", len(notes), discharge.parent
    )
    return ExitCode.OK
