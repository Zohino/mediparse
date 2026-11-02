"""Převod výsledku provenance na návratový kód, společný pro příkazy provenance korpusu a překladu."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, assert_never

from mediparse.application.corpus_provenance import (
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.entrypoints.exit_code import ExitCode

if TYPE_CHECKING:
    from mediparse.application.corpus_provenance import ProvenanceOutcome

logger = logging.getLogger(__name__)


def provenance_exit_code(outcome: ProvenanceOutcome, success_message: str) -> ExitCode:
    """Převede výsledek zápisu provenance na návratový kód a hlášku v logu.

    Args:
        outcome: Výsledek use case provenance.
        success_message: Hláška pro úspěšný zápis.

    Returns:
        OK po zápisu, REFUSED s důvodem odmítnutí v logu.
    """
    match outcome:
        case ProvenanceWritten():
            logger.info("%s", success_message)
            return ExitCode.OK
        case ProvenanceRefused(reason=reason):
            logger.error("%s", reason)
            return ExitCode.REFUSED
        case _:
            assert_never(outcome)
