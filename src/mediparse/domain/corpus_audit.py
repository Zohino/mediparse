"""Audit syntetického korpusu: normalizace textu, sdílené n-gramy s referencí, otisk korpusu a pravidla brány."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
)

from mediparse.domain.note_text import alternation

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

NGRAM_SIZE: Final = 13
NORMALIZATION: Final = "nfkc-lower-alnum-deid-v2"

_TOKEN: Final = re.compile(r"___|[^\W_]+")

type Ngram = tuple[str, ...]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


def commit_sha(value: str) -> str:
    """Ověří, že hodnota je celý SHA-1 hash commitu.

    Returns:
        Hodnota beze změny.

    Raises:
        ValueError: Hodnota není 40 hexadecimálních znaků, například zkrácený hash.
    """
    if _COMMIT.fullmatch(value) is None:
        msg = "Commit musí být celý SHA-1 hash (40 hexadecimálních znaků), například z `git rev-parse HEAD`."
        raise ValueError(msg)
    return value


CommitSha = Annotated[str, AfterValidator(commit_sha)]


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

    def __init__(
        self, notes: Mapping[str, str], structure_labels: Collection[str]
    ) -> None:
        """Zaindexuje n-gramy všech zpráv korpusu, které nepřekročí štítek struktury.

        Štítek je jméno pole preambule, hlavička nebo podnadpis s dvojtečkou, jak je
        zprávě předepisuje config; text mezi štítky se audituje celý.
        """
        pattern = _label_pattern(structure_labels)
        index: defaultdict[Ngram, list[Position]] = defaultdict(list)
        for note, text in notes.items():
            for token, gram in _unlabelled_ngrams(text, pattern):
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
    structure_labels: tuple[str, ...]
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
    return _TOKEN.findall(_normalize(text))


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


def fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """Otisk dvojic jméno a obsah: SHA-256 seznamu ve tvaru výstupu ``sha256sum`` seřazeného podle jména.

    Returns:
        Hexadecimální SHA-256 otisk.
    """
    listing = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(files)
    )
    return hashlib.sha256(listing.encode()).hexdigest()


def audit_record_violations(
    corpus_sha256: str,
    record: AuditRecord | None,
    reference_sha256: Mapping[str, str],
    structure_labels: Collection[str],
) -> tuple[str, ...]:
    """Důvody, proč korpus s daným otiskem neodpovídá svému záznamu auditu; prázdný výsledek znamená, že odpovídá.

    Returns:
        Popisy porušení auditu.
    """
    if record is None:
        return ("Korpus nemá záznam auditu.",)
    violations: list[str] = []
    if record.corpus_sha256 != corpus_sha256:
        violations.append(
            "Korpus se od auditovaného liší, po změně je nutný nový audit."
        )
    if (record.ngram_size, record.normalization) != (NGRAM_SIZE, NORMALIZATION):
        violations.append("Záznam auditu vznikl jinou metodou, je nutný nový audit.")
    if record.structure_labels != tuple(sorted(structure_labels)):
        violations.append(
            "Audit vyřadil jiné štítky struktury, než předepisuje config; je nutný nový audit."
        )
    mismatch = reference_mismatch(
        ((file.name, file.sha256) for file in record.reference), reference_sha256
    )
    if mismatch:
        violations.append(
            f"Audit neběžel přesně proti referenci z configu, liší se: {', '.join(mismatch)}; je nutný nový audit."
        )
    return tuple(violations)


def reference_mismatch(
    files: Iterable[tuple[str, str]], reference_sha256: Mapping[str, str]
) -> tuple[str, ...]:
    """Soubory, kterými se auditovaná reference liší od reference z configu.

    Returns:
        Seřazená jména chybějících, přebývajících a zdvojených souborů i souborů s jiným
        otiskem; prázdný výsledek znamená přesnou shodu.
    """
    audited = Counter(files)
    expected = Counter(reference_sha256.items())
    return tuple(
        sorted({name for name, _ in (audited - expected) + (expected - audited)})
    )


def _ngrams(tokens: Sequence[str]) -> Iterator[Ngram]:
    return (
        tuple(tokens[start : start + NGRAM_SIZE])
        for start in range(len(tokens) - NGRAM_SIZE + 1)
    )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _label_pattern(labels: Collection[str]) -> re.Pattern[str]:
    normalized = {_normalize(label) for label in labels}
    return re.compile(rf"(?<![^\W_])(?:{alternation(normalized)}):")


def _unlabelled_ngrams(
    text: str, labels: re.Pattern[str]
) -> Iterator[tuple[int, Ngram]]:
    normalized = _normalize(text)
    spans = [label.span() for label in labels.finditer(normalized)]
    tokens = list(_TOKEN.finditer(normalized))
    in_label = [
        any(start <= token.start() < end for start, end in spans) for token in tokens
    ]
    return (
        (start, gram)
        for start, gram in enumerate(_ngrams([token[0] for token in tokens]))
        if not any(in_label[start : start + NGRAM_SIZE])
    )
