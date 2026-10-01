"""Use case auditu: syntetický korpus nesmí s MIMIC-IV-Note sdílet jediný n-gram slov ani pacienta.

Audit čte MIMIC, proto odmítne běžet ve veřejném CI i v relaci hostovaného
modelu. Odmítnutí kvůli prostředí a korpusu nastanou dřív, než se sáhne na
referenci; otisky referenčních souborů se ověří proti configu dřív, než se
začne číst jejich obsah.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditRecord,
    NgramIndex,
    ReferenceFile,
    ReferenceMismatchError,
    reference_mismatch,
    scan,
)
from mediparse.domain.note import InvalidNoteIdError, subject_of

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from datetime import datetime

    from mediparse.domain.corpus_audit import Ngram, Position, ReferenceNote

_BLOCKING_VARIABLES: Final = MappingProxyType({
    "CI": "Audit čte MIMIC, a proto ve veřejném CI neběží.",
    "CLAUDECODE": "Audit neběží v relaci Claude Code: jeho výstup by odešel hostovanému modelu.",
})


class SyntheticCorpus(Protocol):
    """Auditovaný korpus: zprávy, jejich otisk a místo pro záznam auditu."""

    def notes(self) -> Mapping[str, str]:
        """Texty zpráv.

        Returns:
            Slovník jméno zprávy v korpusu → text zprávy.
        """

    def note_ids(self) -> Iterable[str]:
        """Note_id všech zpráv korpusu.

        Returns:
            Note_id ve skladbě MIMIC-IV-Note.
        """

    def fingerprint(self) -> str | None:
        """Otisk zpráv korpusu.

        Returns:
            SHA-256 otisk, nebo None, když korpus neobsahuje žádnou zprávu.
        """

    def save_record(self, record: AuditRecord) -> None:
        """Uloží záznam čistého auditu ke korpusu."""


class Reference(Protocol):
    """Referenční soubor MIMIC-IV-Note, který audit čte proudově."""

    @property
    def name(self) -> str:
        """Jméno souboru, pod kterým se reference uvede v záznamu auditu."""

    def notes(self) -> Iterable[ReferenceNote]:
        """Zprávy reference po jedné.

        Returns:
            Pacient a text každé zprávy.
        """

    def sha256(self) -> str:
        """Otisk souboru srovnatelný se ``SHA256SUMS.txt`` na PhysioNetu.

        Returns:
            Hexadecimální SHA-256 otisk.
        """


class Workspace(Protocol):
    """Rozložení vstupů na disku, které audit ověří, než sáhne na referenci."""

    def missing_references(self) -> Sequence[str]:
        """Referenční soubory, které neexistují.

        Returns:
            Cesty chybějících souborů.
        """

    def corpus_in_repository(self) -> bool:
        """Zda korpus leží v git repozitáři.

        Returns:
            True, když se kořen repozitáře korpusu dá určit.
        """

    def report_in_repository(self) -> bool:
        """Zda report shod míří dovnitř repozitáře korpusu.

        Returns:
            True, když by report skončil v repozitáři.
        """


class OverlapReport(Protocol):
    """Report shod mimo repozitář: pozice a kolize subject_id, nikdy text zpráv."""

    def write(
        self, positions: Sequence[Position], colliding_subjects: Collection[str]
    ) -> None:
        """Zapíše pozice sdílených n-gramů a kolidující subject_id."""


@dataclass(frozen=True)
class AuditClean:
    """Korpus s referencí nic nesdílí a záznam auditu je uložený."""

    notes: int
    ngrams: int
    reference_notes: int


@dataclass(frozen=True)
class AuditOverlap:
    """Korpus s referencí sdílí n-gramy nebo pacienty; záznam nevznikl, pozice jsou v reportu."""

    shared_ngrams: int
    colliding_subjects: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class AuditRefused:
    """Audit neproběhl; důvod je určený člověku."""

    reason: str


type AuditOutcome = AuditClean | AuditOverlap | AuditRefused


@dataclass(frozen=True)
class CorpusAudit:
    """Audit syntetického korpusu proti referenci MIMIC-IV-Note."""

    workspace: Workspace
    corpus: SyntheticCorpus
    references: Sequence[Reference]
    report: OverlapReport

    def run(
        self,
        environ: Mapping[str, str],
        tool_commit: str,
        created_at: datetime,
        reference_sha256: Mapping[str, str],
    ) -> AuditOutcome:
        """Porovná korpus s referencí; čistý audit uloží záznam, shoda jen report mimo repozitář.

        Returns:
            Výsledek auditu; při odmítnutí se obsah reference nečte.
        """
        problem = _environment_problem(environ) or _workspace_problem(self.workspace)
        if problem is not None:
            return AuditRefused(problem)
        sha256 = self.corpus.fingerprint()
        if sha256 is None:
            return AuditRefused("Korpus neobsahuje žádnou zprávu.")
        try:
            subjects = frozenset(subject_of(note) for note in self.corpus.note_ids())
            digests = _verified_digests(self.references, reference_sha256)
        except (InvalidNoteIdError, ReferenceMismatchError) as error:
            return AuditRefused(str(error))
        notes = self.corpus.notes()
        index = NgramIndex(notes)
        scans = tuple(
            scan(index, subjects, reference.notes()) for reference in self.references
        )
        if any(result.rows == 0 for result in scans):
            return AuditRefused("Referenční soubor neobsahuje žádnou zprávu.")
        shared = {gram for result in scans for gram in result.shared}
        colliding = {
            subject for result in scans for subject in result.colliding_subjects
        }
        if shared or colliding:
            return _overlap(index, shared, colliding, self.report)
        record = AuditRecord(
            corpus_sha256=sha256,
            corpus_files=len(notes),
            ngram_size=NGRAM_SIZE,
            normalization=NORMALIZATION,
            synthetic_ngrams=len(index),
            reference=tuple(
                ReferenceFile(name=reference.name, sha256=digest, rows=result.rows)
                for reference, digest, result in zip(
                    self.references, digests, scans, strict=True
                )
            ),
            shared_ngrams=0,
            colliding_subjects=0,
            tool_commit=tool_commit,
            created_at=created_at,
        )
        self.corpus.save_record(record)
        return AuditClean(
            notes=len(notes),
            ngrams=len(index),
            reference_notes=sum(result.rows for result in scans),
        )


def _overlap(
    index: NgramIndex, shared: set[Ngram], colliding: set[str], report: OverlapReport
) -> AuditOverlap:
    positions = index.positions(shared)
    report.write(positions, colliding)
    return AuditOverlap(
        shared_ngrams=len(shared),
        colliding_subjects=len(colliding),
        notes=tuple(sorted({position.note for position in positions})),
    )


def _verified_digests(
    references: Sequence[Reference], reference_sha256: Mapping[str, str]
) -> tuple[str, ...]:
    digests = tuple(reference.sha256() for reference in references)
    names = (reference.name for reference in references)
    mismatch = reference_mismatch(zip(names, digests, strict=True), reference_sha256)
    if mismatch:
        msg = f"Referenční soubory nesedí s referencí z configu: {', '.join(mismatch)}."
        raise ReferenceMismatchError(msg)
    return digests


def _environment_problem(environ: Mapping[str, str]) -> str | None:
    return next(
        (reason for name, reason in _BLOCKING_VARIABLES.items() if environ.get(name)),
        None,
    )


def _workspace_problem(workspace: Workspace) -> str | None:
    missing = workspace.missing_references()
    if missing:
        return f"Referenční soubory neexistují: {', '.join(missing)}"
    if not workspace.corpus_in_repository():
        return "Korpus neleží v git repozitáři."
    if workspace.report_in_repository():
        return "Report s pozicemi shod musí ležet mimo repozitář."
    return None
