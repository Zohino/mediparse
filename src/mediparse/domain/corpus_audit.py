"""Audit syntetického korpusu: normalizace textu, sdílené n-gramy s referencí, otisk korpusu a pravidla brány."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

NGRAM_SIZE: Final = 13
NORMALIZATION: Final = "nfkc-lower-alnum-deid-v1"

_TOKEN: Final = re.compile(r"___|[^\W_]+")
_NOTE_ID: Final = re.compile(r"(?P<subject>\d+)-DS-\d+")

type Ngram = tuple[str, ...]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class ReferenceNote(NamedTuple):
    """Zpráva referenčního korpusu: pacient a text."""

    subject_id: str
    text: str


@dataclass(frozen=True, order=True)
class Position:
    """Začátek sdíleného n-gramu v syntetické zprávě, určený pořadím tokenu."""

    note: str
    token: int


@dataclass(frozen=True)
class ScanResult:
    """Výsledek průchodu jedním referenčním souborem."""

    rows: int
    shared: frozenset[Ngram]
    colliding_subjects: frozenset[str]


class NgramIndex:
    """N-gramy syntetického korpusu s pozicemi; v paměti drží jen menší stranu srovnání."""

    def __init__(self, notes: Mapping[str, str]) -> None:
        """Zaindexuje všechny n-gramy všech zpráv korpusu."""
        index: defaultdict[Ngram, list[Position]] = defaultdict(list)
        for note, text in notes.items():
            for token, gram in enumerate(_ngrams(tokenize(text))):
                index[gram].append(Position(note, token))
        self._index = dict(index)

    def __len__(self) -> int:
        """Počet různých n-gramů korpusu.

        Returns:
            Počet klíčů indexu.
        """
        return len(self._index)

    def shared_with(self, text: str) -> set[Ngram]:
        """N-gramy textu, které se vyskytují i v korpusu.

        Returns:
            Sdílené n-gramy; prázdná množina, když text s korpusem nic nesdílí.
        """
        return {gram for gram in _ngrams(tokenize(text)) if gram in self._index}

    def positions(self, grams: Iterable[Ngram]) -> list[Position]:
        """Pozice n-gramů v syntetických zprávách, bez jejich textu.

        Returns:
            Seřazené pozice všech výskytů.
        """
        return sorted(position for gram in grams for position in self._index[gram])


class ReferenceFile(BaseModel):
    """Referenční soubor, proti kterému audit běžel."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    sha256: Sha256
    rows: PositiveInt


class AuditRecord(BaseModel):
    """Záznam čistého auditu; vzniká jen tehdy, když korpus s referencí nic nesdílí."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    corpus_sha256: Sha256
    corpus_files: PositiveInt
    ngram_size: PositiveInt
    normalization: str
    synthetic_ngrams: NonNegativeInt
    reference: Annotated[tuple[ReferenceFile, ...], Field(min_length=1)]
    shared_ngrams: Literal[0]
    colliding_subjects: Literal[0]
    tool_commit: CommitSha
    created_at: AwareDatetime


class InvalidAuditRecordError(ValueError):
    """Záznam auditu neodpovídá schématu, a korpus proto neatestuje."""


def tokenize(text: str) -> list[str]:
    """Rozdělí text na tokeny po normalizaci NFKC a převodu na malá písmena.

    Returns:
        Souvislé úseky písmen a číslic; de-identifikační značka ``___`` je jeden token.
    """
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def scan(
    index: NgramIndex, subjects: frozenset[str], notes: Iterable[ReferenceNote]
) -> ScanResult:
    """Projde referenční zprávy a sbírá n-gramy sdílené s korpusem a kolize subject_id.

    Returns:
        Počet prošlých zpráv, sdílené n-gramy a syntetické subject_id, které reference obsahuje.
    """
    rows = 0
    shared: set[Ngram] = set()
    colliding: set[str] = set()
    for note in notes:
        rows += 1
        shared |= index.shared_with(note.text)
        if note.subject_id in subjects:
            colliding.add(note.subject_id)
    return ScanResult(
        rows=rows, shared=frozenset(shared), colliding_subjects=frozenset(colliding)
    )


def subject_of(note_id: str) -> str:
    """Subject_id pacienta z note_id propouštěcí zprávy ve skladbě MIMIC-IV-Note.

    Returns:
        Identifikátor pacienta.

    Raises:
        ValueError: Jméno neodpovídá skladbě ``subject_id-DS-pořadí``.
    """
    match = _NOTE_ID.fullmatch(note_id)
    if match is None:
        msg = f"Neplatné note_id propouštěcí zprávy: {note_id!r}"
        raise ValueError(msg)
    return match["subject"]


def fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """Otisk korpusu: SHA-256 seznamu ve tvaru výstupu ``sha256sum`` seřazeného podle cesty.

    Returns:
        Hexadecimální SHA-256 otisk.
    """
    listing = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {path}\n"
        for path, content in sorted(files)
    )
    return hashlib.sha256(listing.encode()).hexdigest()


def gate_violations(
    corpus_sha256: str | None, record: AuditRecord | None
) -> tuple[str, ...]:
    """Důvody, proč korpus nesmí do repozitáře; prázdný výsledek znamená, že smí.

    Returns:
        Popisy porušení; neexistující korpus nic neporušuje.
    """
    if corpus_sha256 is None:
        return ()
    if record is None:
        return ("Korpus nemá záznam auditu.",)
    violations: list[str] = []
    if record.corpus_sha256 != corpus_sha256:
        violations.append(
            "Korpus se od auditovaného liší, po změně je nutný nový audit."
        )
    if (record.ngram_size, record.normalization) != (NGRAM_SIZE, NORMALIZATION):
        violations.append("Záznam auditu vznikl jinou metodou, je nutný nový audit.")
    return tuple(violations)


def _ngrams(tokens: Sequence[str]) -> Iterator[Ngram]:
    return (
        tuple(tokens[start : start + NGRAM_SIZE])
        for start in range(len(tokens) - NGRAM_SIZE + 1)
    )
