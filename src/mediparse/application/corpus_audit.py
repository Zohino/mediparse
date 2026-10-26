"""Use case auditu: syntetický korpus nesmí s MIMIC-IV-Note sdílet jediný n-gram slov ani pacienta.

Audit čte MIMIC, proto odmítne běžet ve veřejném CI i v relaci hostovaného
modelu. Odmítnutí kvůli prostředí a korpusu nastanou dřív, než se sáhne na
referenci; otisky referenčních souborů se ověří proti configu dřív, než se
začne číst jejich obsah.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.ports.corpus import FingerprintedCorpus, NoteSource
from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditRecord,
    NgramIndex,
    ReferenceFile,
    scan,
)
from mediparse.domain.data_access import blocked_reason
from mediparse.domain.language import Language
from mediparse.domain.note import InvalidNoteIdError, subject_of

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from datetime import datetime

    from mediparse.domain.corpus_audit import (
        AuditPin,
        Ngram,
        Position,
        ReferenceNote,
        ScanResult,
    )
    from mediparse.domain.identity import SubjectId


class SyntheticCorpus(NoteSource, FingerprintedCorpus, Protocol):
    """Auditovaný korpus: zprávy, jejich otisk a místo pro záznam auditu."""

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

        Raises:
            InvalidInputError: Reference je nečitelná nebo neodpovídá schématu;
                chyba vznikne až při iteraci.
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
        """Zda report shod míří dovnitř libovolného git repozitáře.

        Returns:
            True, když by report skončil v některém repozitáři.
        """


class OverlapReport(Protocol):
    """Report shod mimo repozitář: pozice a kolize subject_id, nikdy text zpráv."""

    def write(
        self, positions: Sequence[Position], colliding_subjects: Collection[SubjectId]
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
        pin: AuditPin,
    ) -> AuditOutcome:
        """Porovná korpus s referencí; čistý audit uloží záznam, shoda jen report mimo repozitář.

        Returns:
            Výsledek auditu; při odmítnutí se obsah reference nečte.
        """
        corpus = self._preflight(environ, pin)
        if isinstance(corpus, AuditRefused):
            return corpus
        index = NgramIndex(corpus.notes, pin.structure_labels)
        scans = self._scan(index, corpus.subjects)
        if isinstance(scans, AuditRefused):
            return scans
        shared, colliding = _shared_ngrams(scans), _colliding_subjects(scans)
        if shared or colliding:
            return _overlap(index, shared, colliding, self.report)
        record = self._record(
            corpus, index, scans, pin, _Issued(tool_commit, created_at)
        )
        self.corpus.save_record(record)
        return AuditClean(
            notes=len(corpus.notes),
            ngrams=len(index),
            reference_notes=sum(result.rows for result in scans),
        )

    def _preflight(
        self, environ: Mapping[str, str], pin: AuditPin
    ) -> _CheckedCorpus | AuditRefused:
        # Pořadí je bezpečnostní: odmítnutí kvůli prostředí a korpusu přijdou dřív, než
        # se sáhne na referenci, a otisk reference (čte celý soubor) dřív než její zprávy.
        problem = blocked_reason(environ) or _workspace_problem(self.workspace)
        if problem is not None:
            return AuditRefused(problem)
        sha256 = self.corpus.fingerprint()
        if sha256 is None:
            return AuditRefused("Korpus neobsahuje žádnou zprávu.")
        # Všechny jazyky, protože otisk pokrývá i překlad; audit jen angličtiny by bránou
        # pustil neauditovaný český text. Jazyk v popisku rozliší stejný note_id v en a cs.
        by_language = {language: self.corpus.notes(language) for language in Language}
        problem = _note_id_problem(by_language) or _reference_problem(
            self.references, pin
        )
        if problem is not None:
            return AuditRefused(problem)
        return _CheckedCorpus(
            sha256=sha256,
            notes={
                f"{language}/{note_id}": text
                for language, texts in by_language.items()
                for note_id, text in texts.items()
            },
            subjects=frozenset(
                subject_of(note_id)
                for texts in by_language.values()
                for note_id in texts
            ),
        )

    def _scan(
        self, index: NgramIndex, subjects: frozenset[SubjectId]
    ) -> tuple[ScanResult, ...] | AuditRefused:
        scans = tuple(
            scan(index, subjects, reference.notes()) for reference in self.references
        )
        if any(result.rows == 0 for result in scans):
            return AuditRefused("Referenční soubor neobsahuje žádnou zprávu.")
        return scans

    def _record(
        self,
        corpus: _CheckedCorpus,
        index: NgramIndex,
        scans: Sequence[ScanResult],
        pin: AuditPin,
        issued: _Issued,
    ) -> AuditRecord:
        return AuditRecord(
            corpus_sha256=corpus.sha256,
            corpus_files=len(corpus.notes),
            ngram_size=NGRAM_SIZE,
            normalization=NORMALIZATION,
            structure_labels=tuple(sorted(pin.structure_labels)),
            synthetic_ngrams=len(index),
            reference=tuple(
                ReferenceFile(
                    name=reference.name,
                    sha256=pin.reference_sha256[reference.name],
                    rows=result.rows,
                )
                for reference, result in zip(self.references, scans, strict=True)
            ),
            shared_ngrams=0,
            colliding_subjects=0,
            tool_commit=issued.tool_commit,
            created_at=issued.created_at,
        )


@dataclass(frozen=True)
class _CheckedCorpus:
    """Korpus, který prošel předběžnými kontrolami: otisk, zprávy podle ``jazyk/note_id`` a jejich pacienti."""

    sha256: str
    notes: Mapping[str, str]
    subjects: frozenset[SubjectId]


@dataclass(frozen=True)
class _Issued:
    """Kdo a kdy záznam auditu vydal: commit nástroje a čas běhu."""

    tool_commit: str
    created_at: datetime


def _shared_ngrams(scans: Iterable[ScanResult]) -> set[Ngram]:
    return {gram for result in scans for gram in result.shared}


def _colliding_subjects(scans: Iterable[ScanResult]) -> set[SubjectId]:
    return {subject for result in scans for subject in result.colliding_subjects}


def _overlap(
    index: NgramIndex,
    shared: set[Ngram],
    colliding: set[SubjectId],
    report: OverlapReport,
) -> AuditOverlap:
    positions = index.positions(shared)
    report.write(positions, colliding)
    return AuditOverlap(
        shared_ngrams=len(shared),
        colliding_subjects=len(colliding),
        notes=tuple(sorted({position.note for position in positions})),
    )


def _note_id_problem(by_language: Mapping[Language, Mapping[str, str]]) -> str | None:
    try:
        for texts in by_language.values():
            for note_id in texts:
                subject_of(note_id)
    except InvalidNoteIdError as error:
        return str(error)
    return None


def _reference_problem(references: Sequence[Reference], pin: AuditPin) -> str | None:
    mismatch = pin.reference_mismatch(
        (reference.name, reference.sha256()) for reference in references
    )
    if mismatch:
        return (
            f"Referenční soubory nesedí s referencí z configu: {', '.join(mismatch)}."
        )
    return None


def _workspace_problem(workspace: Workspace) -> str | None:
    missing = workspace.missing_references()
    if missing:
        return f"Referenční soubory neexistují: {', '.join(missing)}"
    if not workspace.corpus_in_repository():
        return "Korpus neleží v git repozitáři."
    if workspace.report_in_repository():
        return "Report s pozicemi shod musí ležet mimo jakýkoli git repozitář."
    return None
