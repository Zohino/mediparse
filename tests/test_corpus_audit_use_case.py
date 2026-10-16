"""Use case auditu nad fakes: pořadí odmítnutí, výsledek a to, že se při odmítnutí nesáhne na MIMIC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mediparse.application.corpus_audit import (
    AuditClean,
    AuditOverlap,
    AuditRefused,
    CorpusAudit,
)
from mediparse.domain.corpus_audit import ReferenceNote
from mediparse.domain.language import Language
from tests.support import (
    COMMIT,
    CORPUS_SHA,
    NOTE_ID,
    PINNED,
    SENTENCE,
    SHORT_TEXT,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

    from mediparse.domain.corpus_audit import AuditRecord, Position
    from mediparse.domain.identity import SubjectId

CREATED_AT = datetime(2026, 9, 29, tzinfo=UTC)
DISCHARGE_SHA = PINNED["discharge.csv.gz"]
DISCHARGE_PINNED = {"discharge.csv.gz": DISCHARGE_SHA}
STRUCTURE_LABELS = frozenset({"Name", "Unit No"})
BOTH_LANGUAGES = 2
SHORT_NOTES = {Language.EN: {NOTE_ID: SHORT_TEXT}}
PREAMBLE = "Name: ___ Unit No: ___ Name: ___ Unit No: ___ Name: ___ Unit No: ___\n"


@dataclass
class _Corpus:
    texts: dict[Language, dict[str, str]]
    saved: list[AuditRecord] = field(default_factory=list)

    def notes(self, language: Language) -> dict[str, str]:
        return self.texts.get(language, {})

    def fingerprint(self) -> str | None:
        return CORPUS_SHA if any(self.texts.values()) else None

    def save_record(self, record: AuditRecord) -> None:
        self.saved.append(record)


@dataclass(frozen=True)
class _Reference:
    rows: tuple[ReferenceNote, ...]
    name: str = "discharge.csv.gz"
    digest: str = DISCHARGE_SHA
    hashed: list[str] = field(default_factory=list)

    def notes(self) -> Iterator[ReferenceNote]:
        return iter(self.rows)

    def sha256(self) -> str:
        self.hashed.append(self.name)
        return self.digest


@dataclass(frozen=True)
class _UnreadableReference:
    name: str = "discharge.csv.gz"
    digest: str = DISCHARGE_SHA

    def notes(self) -> Iterator[ReferenceNote]:
        pytest.fail(
            f"Audit četl zprávy {self.name}, ačkoli měl odmítnout podle otisku."
        )

    def sha256(self) -> str:
        return self.digest


@dataclass(frozen=True)
class _UntouchableReference:
    name: str = "discharge.csv.gz"

    def notes(self) -> Iterator[ReferenceNote]:
        pytest.fail(f"Audit četl {self.name}, ačkoli měl odmítnout.")

    def sha256(self) -> str:
        pytest.fail(f"Audit četl {self.name}, ačkoli měl odmítnout.")


@dataclass(frozen=True)
class _Workspace:
    missing: tuple[str, ...] = ()
    corpus_in_repo: bool = True
    report_in_repo: bool = False

    def missing_references(self) -> tuple[str, ...]:
        return self.missing

    def corpus_in_repository(self) -> bool:
        return self.corpus_in_repo

    def report_in_repository(self) -> bool:
        return self.report_in_repo


@dataclass
class _Report:
    written: list[tuple[tuple[Position, ...], frozenset[SubjectId]]] = field(
        default_factory=list
    )

    def write(
        self, positions: Sequence[Position], colliding_subjects: Collection[SubjectId]
    ) -> None:
        self.written.append((tuple(positions), frozenset(colliding_subjects)))


def _audit(
    corpus: _Corpus,
    reference: _Reference | _UntouchableReference | _UnreadableReference,
    report: _Report,
    workspace: _Workspace | None = None,
) -> CorpusAudit:
    return CorpusAudit(
        workspace=workspace or _Workspace(),
        corpus=corpus,
        references=(reference,),
        report=report,
        structure_labels=STRUCTURE_LABELS,
    )


def test_clean_corpus_saves_record() -> None:
    """Čistý korpus dostane záznam s otiskem, metodou a referencí, report nevznikne."""
    corpus = _Corpus(dict(SHORT_NOTES))
    report = _Report()
    reference = _Reference((ReferenceNote(10000032, SENTENCE),))

    outcome = _audit(corpus, reference, report).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert outcome == AuditClean(notes=1, ngrams=0, reference_notes=1)
    (record,) = corpus.saved
    assert record.corpus_sha256 == CORPUS_SHA
    assert record.reference[0].name == "discharge.csv.gz"
    assert record.reference[0].sha256 == DISCHARGE_SHA
    assert reference.hashed == ["discharge.csv.gz"]
    assert record.created_at == CREATED_AT
    assert not report.written


def test_overlap_writes_report_and_no_record() -> None:
    """Shoda zapíše jen pozice do reportu a záznam auditu nevznikne."""
    corpus = _Corpus({Language.EN: {NOTE_ID: SENTENCE}})
    report = _Report()
    reference = _Reference((ReferenceNote(10000032, SENTENCE),))

    outcome = _audit(corpus, reference, report).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert outcome == AuditOverlap(
        shared_ngrams=3, colliding_subjects=0, notes=(f"en/{NOTE_ID}",)
    )
    assert not corpus.saved
    ((positions, colliding),) = report.written
    assert {position.note for position in positions} == {f"en/{NOTE_ID}"}
    assert not colliding


def test_subject_collision_is_overlap() -> None:
    """Syntetický pacient, který existuje v referenci, audit neprojde."""
    corpus = _Corpus(dict(SHORT_NOTES))
    report = _Report()
    reference = _Reference((ReferenceNote(90000001, "unrelated text"),))

    outcome = _audit(corpus, reference, report).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert outcome == AuditOverlap(shared_ngrams=0, colliding_subjects=1, notes=())
    assert not corpus.saved


@pytest.mark.parametrize(
    "environ", [{"CI": "true"}, {"CLAUDECODE": "1"}], ids=["ci", "claude-code"]
)
def test_environment_refusal_never_touches_reference(environ: dict[str, str]) -> None:
    """V CI ani v relaci Claude Code audit na MIMIC nesáhne."""
    corpus = _Corpus({Language.EN: {NOTE_ID: SENTENCE}})
    report = _Report()

    outcome = _audit(corpus, _UntouchableReference(), report).run(
        environ, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert isinstance(outcome, AuditRefused)
    assert not corpus.saved
    assert not report.written


@pytest.mark.parametrize(
    "workspace",
    [
        _Workspace(missing=("discharge.csv.gz",)),
        _Workspace(corpus_in_repo=False),
        _Workspace(report_in_repo=True),
    ],
    ids=["missing-reference", "corpus-outside-repository", "report-inside-repository"],
)
def test_workspace_refusal_never_touches_reference(workspace: _Workspace) -> None:
    """Chybějící reference, korpus mimo repozitář i report v repozitáři audit odmítne předem."""
    corpus = _Corpus({Language.EN: {NOTE_ID: SENTENCE}})

    outcome = _audit(corpus, _UntouchableReference(), _Report(), workspace).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert isinstance(outcome, AuditRefused)
    assert not corpus.saved


def test_invalid_note_id_is_refused_before_reference() -> None:
    """Soubor, jehož jméno není note_id, audit odmítne dřív, než sáhne na MIMIC."""
    corpus = _Corpus({Language.EN: {"not-a-note": SENTENCE}})

    outcome = _audit(corpus, _UntouchableReference(), _Report()).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert isinstance(outcome, AuditRefused)
    assert "note_id" in outcome.reason


def test_empty_corpus_is_refused() -> None:
    """Prázdný korpus nemá co auditovat."""
    outcome = _audit(_Corpus({}), _UntouchableReference(), _Report()).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert isinstance(outcome, AuditRefused)


def test_empty_reference_is_refused_without_record() -> None:
    """Reference bez jediné zprávy by atestovala nic, záznam nevznikne."""
    corpus = _Corpus(dict(SHORT_NOTES))

    outcome = _audit(corpus, _Reference(()), _Report()).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert isinstance(outcome, AuditRefused)
    assert not corpus.saved


@pytest.mark.parametrize(
    ("reference", "pinned", "named"),
    [
        pytest.param(
            _UnreadableReference(digest="d" * 64),
            DISCHARGE_PINNED,
            "discharge.csv.gz",
            id="jiny-otisk",
        ),
        pytest.param(
            _UnreadableReference(),
            DISCHARGE_PINNED | {"radiology.csv.gz": "e" * 64},
            "radiology.csv.gz",
            id="bez-radiology",
        ),
    ],
)
def test_reference_other_than_config_is_refused_before_reading(
    reference: _UnreadableReference, pinned: dict[str, str], named: str
) -> None:
    """Reference, která nesedí s configem, se odmítne podle otisku dřív, než se čtou zprávy."""
    corpus = _Corpus(dict(SHORT_NOTES))

    outcome = _audit(corpus, reference, _Report()).run({}, COMMIT, CREATED_AT, pinned)

    assert isinstance(outcome, AuditRefused)
    assert named in outcome.reason
    assert not corpus.saved


def test_structure_labels_reach_the_index() -> None:
    """Štítky struktury dojdou do indexu: samotná struktura shodou není."""
    reference = _Reference((ReferenceNote(10000032, f"{PREAMBLE}{SENTENCE}"),))

    outcome = _audit(
        _Corpus({Language.EN: {NOTE_ID: f"{PREAMBLE}a short body"}}),
        reference,
        _Report(),
    ).run({}, COMMIT, CREATED_AT, DISCHARGE_PINNED)

    assert isinstance(outcome, AuditClean)


def test_czech_notes_are_indexed() -> None:
    """N-gram sdílený jen s českou zprávou je shoda; audit nepřestane číst češtinu."""
    corpus = _Corpus({
        Language.EN: {NOTE_ID: SHORT_TEXT},
        Language.CS: {NOTE_ID: SENTENCE},
    })
    report = _Report()
    reference = _Reference((ReferenceNote(10000032, SENTENCE),))

    outcome = _audit(corpus, reference, report).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert outcome == AuditOverlap(
        shared_ngrams=3, colliding_subjects=0, notes=(f"cs/{NOTE_ID}",)
    )
    assert not corpus.saved


def test_same_note_id_in_both_languages_has_two_positions() -> None:
    """Stejné note_id v en i cs se v reportu rozliší jazykem."""
    corpus = _Corpus({
        Language.EN: {NOTE_ID: SENTENCE},
        Language.CS: {NOTE_ID: SENTENCE},
    })
    report = _Report()
    reference = _Reference((ReferenceNote(10000032, SENTENCE),))

    outcome = _audit(corpus, reference, report).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert isinstance(outcome, AuditOverlap)
    assert outcome.notes == (f"cs/{NOTE_ID}", f"en/{NOTE_ID}")
    ((positions, _),) = report.written
    assert {position.note for position in positions} == {
        f"cs/{NOTE_ID}",
        f"en/{NOTE_ID}",
    }


def test_corpus_files_counts_notes_of_all_languages() -> None:
    """Záznam auditu počítá zprávy obou jazyků."""
    corpus = _Corpus({
        Language.EN: {NOTE_ID: SHORT_TEXT},
        Language.CS: {NOTE_ID: "krátká syntetická zpráva"},
    })

    outcome = _audit(
        corpus, _Reference((ReferenceNote(10000032, "x"),)), _Report()
    ).run({}, COMMIT, CREATED_AT, DISCHARGE_PINNED)

    assert outcome == AuditClean(notes=BOTH_LANGUAGES, ngrams=0, reference_notes=1)
    (record,) = corpus.saved
    assert record.corpus_files == BOTH_LANGUAGES


def test_subjects_come_from_every_language() -> None:
    """Pacient, který je jen v české zprávě, se s referencí také srovná."""
    corpus = _Corpus({
        Language.EN: {NOTE_ID: SHORT_TEXT},
        Language.CS: {"90000002-DS-1": "krátká zpráva"},
    })
    reference = _Reference((ReferenceNote(90000002, "unrelated text"),))

    outcome = _audit(corpus, reference, _Report()).run(
        {}, COMMIT, CREATED_AT, DISCHARGE_PINNED
    )

    assert outcome == AuditOverlap(shared_ngrams=0, colliding_subjects=1, notes=())
