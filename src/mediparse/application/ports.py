"""Porty, které používá víc use casů."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.corpus_audit import AuditRecord
    from mediparse.domain.note_plan import NotePlan


class PlanSource(Protocol):
    """Plány zpráv korpusu."""

    def load(self) -> tuple[NotePlan, ...]:
        """Načte plány.

        Returns:
            Plány v pořadí zdroje.

        Raises:
            InvalidInputError: Zdroj plánů chybí nebo neodpovídá schématu; use case
                chybu propouští ke vstupnímu bodu.
        """


class NoteSource(Protocol):
    """Texty zpráv korpusu."""

    def notes(self) -> Mapping[str, str]:
        """Texty zpráv.

        Returns:
            Slovník relativní cesta ``<jazyk>/<note_id>.txt`` → text zprávy.
        """


class FingerprintedCorpus(Protocol):
    """Korpus s otiskem zpráv."""

    def fingerprint(self) -> str | None:
        """Otisk zpráv korpusu.

        Returns:
            SHA-256 otisk, nebo None, když korpus neobsahuje žádnou zprávu.
        """


class AuditedCorpus(FingerprintedCorpus, Protocol):
    """Korpus se záznamem auditu: otisk zpráv, záznam auditu a otisk jeho souboru."""

    def audit_record(self) -> AuditRecord | None:
        """Záznam posledního auditu.

        Returns:
            Záznam, nebo None, když chybí.

        Raises:
            InvalidAuditRecordError: Záznam neodpovídá schématu.
        """

    def audit_sha256(self) -> str:
        """Otisk souboru se záznamem auditu; záznam musí existovat.

        Returns:
            SHA-256 otisk.
        """
