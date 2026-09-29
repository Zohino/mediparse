"""Vstupní bod brány syntetického korpusu: do repozitáře smí jen korpus, jehož otisk sedí se záznamem auditu.

Konzolový skript ``mediparse-corpus-gate`` běží jako hook prek při commitu i pushi
a jako test v CI. S daty MIMIC nepracuje, proto smí běžet i ve veřejném CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import InvalidAuditRecordError, gate_violations
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_corpus import (
    CORPUS_ROOT,
    corpus_sha256,
    load_record,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-gate``.

    Returns:
        Návratový kód brány.
    """
    return run(sys.argv[1:])


def run(argv: Sequence[str]) -> ExitCode:
    """Porovná otisk korpusu se záznamem auditu.

    Returns:
        OK, když korpus neexistuje nebo odpovídá auditu, jinak BLOCKED.
    """
    corpus = _parser().parse_args(argv).corpus
    try:
        record = load_record(corpus)
    except InvalidAuditRecordError:
        violations: tuple[str, ...] = ("Záznam auditu neodpovídá schématu.",)
    else:
        violations = gate_violations(corpus_sha256(corpus), record)
    for violation in violations:
        sys.stderr.write(f"{violation}\n")
    return ExitCode.BLOCKED if violations else ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-gate",
        description="Brána: korpus v repozitáři odpovídá svému auditu.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    return parser
