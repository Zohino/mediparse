"""Vstupní bod kroku Snakemake ``build_cohort``: kohorta zpráv se štítky pětice a její manifest.

Cesty určuje pravidlo; skript workflow jen rozbalí objekt ``snakemake`` a zavolá
``main``. Diagnóza pod deset pozitivních nebo negativních zpráv a prázdná kohorta
ukončí krok kódem REFUSED dřív, než vznikne soubor kohorty nebo manifest.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.application.cohort import CohortMaterialization
from mediparse.domain.cohort import CohortManifest
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.cohort_manifest_file import JsonCohortManifestFile
from mediparse.infrastructure.diagnosis_codes_file import load_diagnosis_codes
from mediparse.infrastructure.duckdb_cohort import DuckDbCohort
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.mimic_tables import load_mimic_tables
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.source_revision import revision_source

if TYPE_CHECKING:
    from collections.abc import Callable

    from mediparse.domain.run_manifest import SourceRevision

logger = logging.getLogger(__name__)

PACKAGES: Final = ("python", "duckdb", "pyarrow")
SOURCE_TABLES: Final = ("discharge.csv.gz", "diagnoses_icd.csv.gz")
SPILL_DIRECTORY: Final = "duckdb-tmp"


def main(
    *,
    discharge: Path,
    diagnoses: Path,
    candidates: Path,
    tables: Path,
    cohort: Path,
    manifest: Path,
    log: Path,
) -> ExitCode:
    """Krok pravidla: diagnostika do souboru logu pravidla.

    Returns:
        Návratový kód kroku.
    """
    configure_logging(log)
    return run(
        discharge=discharge,
        diagnoses=diagnoses,
        candidates=candidates,
        tables=tables,
        cohort=cohort,
        manifest=manifest,
        revision=revision_source(os.environ, Path.cwd()),
    )


def _source_digests(tables: Path) -> dict[str, str]:
    pinned = load_mimic_tables(tables)
    missing = [name for name in SOURCE_TABLES if name not in pinned]
    if missing:
        msg = f"V configu tabulek {tables} chybí: {', '.join(missing)}."
        raise InvalidInputError(msg)
    return {name: pinned[name].sha256 for name in SOURCE_TABLES}


@refusing_invalid_input
def run(
    *,
    discharge: Path,
    diagnoses: Path,
    candidates: Path,
    tables: Path,
    cohort: Path,
    manifest: Path,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    """Složí use case z tabulek, configu a adaptéru DuckDB a zapíše kohortu s manifestem.

    Returns:
        OK po zápisu kohorty a manifestu; REFUSED pro chybějící nebo neplatný vstup,
        nečitelnou revizi zdroje, diagnózu pod deset zpráv a prázdnou kohortu.
    """
    codes = load_diagnosis_codes(candidates)
    sources = _source_digests(tables)
    source = revision()
    result = CohortMaterialization(
        DuckDbCohort(discharge, diagnoses, cohort, cohort.parent / SPILL_DIRECTORY),
        codes,
    ).run()
    JsonCohortManifestFile(manifest).write(
        CohortManifest(
            cohort_sha256=file_sha256(cohort),
            config_sha256=file_sha256(candidates),
            sources=sources,
            packages=installed_versions(PACKAGES),
            source=source,
            counts=result.counts,
            invalid_codes=result.invalid_codes,
        )
    )
    logger.info(
        "Kohorta: %d zpráv a %d pacientů v %s, neplatných kódů %d.",
        result.counts.funnel[-1].notes,
        result.counts.funnel[-1].patients,
        cohort,
        result.invalid_codes,
    )
    return ExitCode.OK
