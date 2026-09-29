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
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditRecord,
    NgramIndex,
    ReferenceFile,
    scan,
    subject_of,
)
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_reference import file_sha256, reference_notes
from mediparse.infrastructure.overlap_report import write_overlap_report
from mediparse.infrastructure.synthetic_corpus import (
    CORPUS_ROOT,
    corpus_sha256,
    read_notes,
    save_record,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.corpus_audit import Ngram, ScanResult

_BLOCKING_VARIABLES: Final = MappingProxyType({
    "CI": "Audit čte MIMIC, a proto ve veřejném CI neběží.",
    "CLAUDECODE": "Audit neběží v relaci Claude Code: jeho výstup by odešel hostovanému modelu.",
})
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-audit``.

    Returns:
        Návratový kód auditu.
    """
    return run(sys.argv[1:], os.environ)


def run(argv: Sequence[str], environ: Mapping[str, str]) -> ExitCode:
    """Spustí audit s argumenty z příkazové řádky.

    Returns:
        Výsledek auditu.
    """
    args = _parser().parse_args(argv)
    return run_audit(args.corpus, args.reference, args.report, args.commit, environ)


def run_audit(
    corpus: Path,
    references: Sequence[Path],
    report: Path,
    commit: str,
    environ: Mapping[str, str],
) -> ExitCode:
    """Porovná korpus s referencí; čistý audit zapíše záznam, shoda jen report mimo repozitář.

    Odmítne běžet ve veřejném CI, v relaci Claude Code a s reportem uvnitř
    repozitáře, protože čte MIMIC a jeho výstup nesmí opustit kontrolované prostředí.

    Returns:
        OK bez shod a kolizí subject_id, BLOCKED při shodě, REFUSED při odmítnutí.
    """
    problem = _environment_problem(environ) or _input_problem(
        corpus, references, report
    )
    if problem is not None:
        return _refuse(problem)
    sha256 = corpus_sha256(corpus)
    if sha256 is None:
        return _refuse("Korpus neobsahuje žádnou zprávu.")
    notes = read_notes(corpus)
    index = NgramIndex(notes)
    scans = _scan_references(index, notes, references)
    if any(result.rows == 0 for result in scans.values()):
        return _refuse("Referenční soubor neobsahuje žádnou zprávu.")
    shared = {gram for result in scans.values() for gram in result.shared}
    colliding = {
        subject for result in scans.values() for subject in result.colliding_subjects
    }
    if shared or colliding:
        return _report_overlap(index, shared, colliding, report)
    save_record(corpus, _record(sha256, len(notes), len(index), scans, commit))
    rows = sum(result.rows for result in scans.values())
    _say(
        f"Audit čistý: {len(notes)} zpráv, {len(index)} n-gramů, {rows} referenčních zpráv."
    )
    return ExitCode.OK


def _scan_references(
    index: NgramIndex, notes: Mapping[str, str], references: Sequence[Path]
) -> dict[Path, ScanResult]:
    subjects = frozenset(subject_of(Path(note).stem) for note in notes)
    return {path: scan(index, subjects, reference_notes(path)) for path in references}


def _report_overlap(
    index: NgramIndex, shared: set[Ngram], colliding: set[str], report: Path
) -> ExitCode:
    positions = index.positions(shared)
    write_overlap_report(report, positions, colliding)
    notes = sorted({position.note for position in positions})
    _say(
        f"Sdílené {NGRAM_SIZE}-gramy: {len(shared)}, kolize subject_id: {len(colliding)}."
    )
    _say(f"Zprávy k přegenerování: {', '.join(notes) or 'žádné'}")
    _say(f"Pozice shod: {report}")
    return ExitCode.BLOCKED


def _record(
    sha256: str, files: int, ngrams: int, scans: Mapping[Path, ScanResult], commit: str
) -> AuditRecord:
    reference = tuple(
        ReferenceFile(name=path.name, sha256=file_sha256(path), rows=result.rows)
        for path, result in scans.items()
    )
    return AuditRecord(
        corpus_sha256=sha256,
        corpus_files=files,
        ngram_size=NGRAM_SIZE,
        normalization=NORMALIZATION,
        synthetic_ngrams=ngrams,
        reference=reference,
        shared_ngrams=0,
        colliding_subjects=0,
        tool_commit=commit,
        created_at=datetime.now(tz=UTC),
    )


def _environment_problem(environ: Mapping[str, str]) -> str | None:
    return next(
        (reason for name, reason in _BLOCKING_VARIABLES.items() if environ.get(name)),
        None,
    )


def _input_problem(
    corpus: Path, references: Sequence[Path], report: Path
) -> str | None:
    missing = [str(path) for path in references if not path.is_file()]
    if missing:
        return f"Referenční soubory neexistují: {', '.join(missing)}"
    repository = _repository_root(corpus)
    if repository is None:
        return "Korpus neleží v git repozitáři."
    if report.resolve().is_relative_to(repository):
        return "Report s pozicemi shod musí ležet mimo repozitář."
    return None


def _repository_root(path: Path) -> Path | None:
    resolved = path.resolve()
    return next(
        (
            folder
            for folder in (resolved, *resolved.parents)
            if (folder / ".git").exists()
        ),
        None,
    )


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


def _refuse(reason: str) -> ExitCode:
    sys.stderr.write(f"{reason}\n")
    return ExitCode.REFUSED
