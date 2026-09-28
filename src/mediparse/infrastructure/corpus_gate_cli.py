"""Brána syntetického korpusu: do repozitáře smí jen korpus, jehož otisk sedí se záznamem auditu.

Běží jako hook prek při commitu i pushi a jako test v CI. S daty MIMIC nepracuje,
proto smí běžet i ve veřejném CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from mediparse.domain.corpus_audit import gate_violations
from mediparse.infrastructure.corpus_audit_cli import Exit
from mediparse.infrastructure.synthetic_corpus import (
    CORPUS_ROOT,
    corpus_sha256,
    load_record,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str]) -> Exit:
    """Porovná otisk korpusu se záznamem auditu.

    Returns:
        CLEAN, když korpus neexistuje nebo odpovídá auditu, jinak FOUND.
    """
    corpus = _parser().parse_args(argv).corpus
    try:
        record = load_record(corpus)
    except ValidationError:
        violations: tuple[str, ...] = ("Záznam auditu neodpovídá schématu.",)
    else:
        violations = gate_violations(corpus_sha256(corpus), record)
    for violation in violations:
        sys.stderr.write(f"{violation}\n")
    return Exit.FOUND if violations else Exit.CLEAN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Brána: korpus v repozitáři odpovídá svému auditu."
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
