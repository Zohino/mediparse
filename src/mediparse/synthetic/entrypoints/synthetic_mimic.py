"""Vstupní bod syntetických tabulek MIMIC pro kohortu smoketestu."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.language import Language
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker
from mediparse.infrastructure.table_provenance import write_table_provenance
from mediparse.synthetic.domain.synthetic_mimic import input_notes, synthetic_mimic_rows
from mediparse.synthetic.infrastructure.plans_file import PlansFile
from mediparse.synthetic.infrastructure.synthetic_corpus import CorpusDirectory

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class SyntheticMimicPaths:
    """Vstupní a výstupní cesty tvorby syntetických tabulek MIMIC."""

    plans: Path
    corpus: Path
    candidates: Path
    manifest: Path
    discharge: Path
    diagnoses: Path
    artifacts: Path


def main(*, paths: SyntheticMimicPaths, log: Path) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(paths=paths)


@refusing_invalid_input
def run(*, paths: SyntheticMimicPaths) -> ExitCode:
    """Zapíše tabulky, jejich otisky a značky syntetiky tabulek i odvozených artefaktů.

    Returns:
        OK po zápisu; REFUSED pro neplatné vstupy nebo nespárované plány a zprávy.
    """
    notes = input_notes(
        PlansFile(paths.plans).load(),
        CorpusDirectory(paths.corpus).notes(Language.EN),
        Language.EN,
    )
    shapes = read_manifest(paths.manifest)
    rows = synthetic_mimic_rows(notes, load_diagnosis_codes(paths.candidates), shapes)
    outputs = {"discharge": paths.discharge, "diagnoses_icd": paths.diagnoses}
    for table, path in outputs.items():
        write_string_table(path, shapes[table].columns, rows[table])
        # Syntetika nemá CSV; .invalid odliší její původ od skutečných tabulek MIMIC.
        write_table_provenance(
            path,
            source=f"https://mediparse.invalid/synthetic/{table}",
            source_sha256=file_sha256(path),
        )
    marker = SyntheticMarker(prefix=None, rows=len(notes))
    write_marker(paths.discharge.parent, marker)
    write_marker(paths.artifacts, marker)
    logger.info(
        "Syntetické tabulky MIMIC: %d zpráv v %s.", len(notes), paths.discharge.parent
    )
    return ExitCode.OK
