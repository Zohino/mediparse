"""Vstupní bod lokálního auditu syntetického korpusu proti volnému textu MIMIC-IV-Note.

Konzolový skript ``mediparse-corpus-audit`` spouští jen autor v kontrolovaném
prostředí, mimo veřejné CI i mimo relaci hostovaného modelu. Na výstup jde jen
počet shod a note_id dotčených syntetických zpráv. Pozice shod jdou do reportu
mimo repozitář a shodný text se nevypisuje nikdy.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

from mediparse.application.corpus_audit import (
    AuditClean,
    AuditOverlap,
    AuditRefused,
    CorpusAudit,
)
from mediparse.domain.corpus_audit import NGRAM_SIZE
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.audit_workspace import LocalWorkspace
from mediparse.infrastructure.mimic_reference import MimicReference
from mediparse.infrastructure.mimic_tables import TABLES_PATH, load_reference_sha256
from mediparse.infrastructure.overlap_report import OverlapReportFile
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT, CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.application.corpus_audit import AuditOutcome

_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-audit``.

    Returns:
        Návratový kód auditu.
    """
    return run(sys.argv[1:], os.environ)


def run(argv: Sequence[str], environ: Mapping[str, str]) -> ExitCode:
    """Složí audit z adaptérů podle argumentů a přeloží jeho výsledek na návratový kód.

    Returns:
        OK bez shod, BLOCKED při shodě, REFUSED při odmítnutí.
    """
    args = _parser().parse_args(argv)
    references = tuple(args.reference)
    audit = CorpusAudit(
        workspace=LocalWorkspace(args.corpus, references, args.report),
        corpus=CorpusDirectory(args.corpus),
        references=tuple(MimicReference(path) for path in references),
        report=OverlapReportFile(args.report),
    )
    outcome = audit.run(
        environ, args.commit, datetime.now(tz=UTC), load_reference_sha256(args.tables)
    )
    return _present(outcome, args.report)


def _present(outcome: AuditOutcome, report: Path) -> ExitCode:
    match outcome:
        case AuditClean(notes=notes, ngrams=ngrams, reference_notes=rows):
            _say(
                f"Audit čistý: {notes} zpráv, {ngrams} n-gramů, {rows} referenčních zpráv."
            )
            return ExitCode.OK
        case AuditOverlap(
            shared_ngrams=shared, colliding_subjects=colliding, notes=notes
        ):
            _say(
                f"Sdílené {NGRAM_SIZE}-gramy: {shared}, kolize subject_id: {colliding}."
            )
            _say(f"Zprávy k přegenerování: {', '.join(notes) or 'žádné'}")
            _say(f"Pozice shod: {report}")
            return ExitCode.BLOCKED
        case AuditRefused(reason=reason):
            sys.stderr.write(f"{reason}\n")
            return ExitCode.REFUSED
        case _:
            assert_never(outcome)


def _commit(value: str) -> str:
    if _COMMIT.fullmatch(value) is None:
        msg = "Commit nástroje musí být celý SHA-1 hash, například z `git rev-parse HEAD`."
        raise argparse.ArgumentTypeError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-audit",
        description="Audit syntetického korpusu proti volnému textu MIMIC-IV-Note.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        required=True,
        help="discharge.csv.gz nebo radiology.csv.gz z MIMIC-IV-Note; opakovatelný",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLES_PATH,
        help="config tabulek MIMIC, z něhož se bere reference a její otisky",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="soubor s pozicemi shod mimo repozitář",
    )
    parser.add_argument(
        "--commit",
        type=_commit,
        required=True,
        help="commit nástroje, `git rev-parse HEAD`",
    )
    return parser


def _say(message: str) -> None:
    sys.stdout.write(f"{message}\n")
