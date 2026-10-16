"""Use case provenance překladu nad fakes: vazba na korpus, audit a zápis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.corpus_provenance import (
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.application.translation_provenance import TranslationProvenance
from mediparse.domain.corpus_provenance import (
    Generation,
    ProvenanceRecord,
    Translation,
)
from mediparse.domain.language import Language
from mediparse.domain.translation_consistency import TranslatedNote
from tests.support import (
    CORPUS_SHA,
    PINNED,
    STRUCTURE_LABELS,
    audit_record,
    translation_record,
)
from tests.workdir_support import sha256_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from mediparse.domain.corpus_audit import AuditRecord

NEW_AUDIT_FILE = "8" * 64
OLD_AUDIT_FILE = "9" * 64


def _generation() -> Generation:
    return Generation.model_validate_json(
        json.dumps({
            "seed": 0,
            "sampler_config_sha256": "1" * 64,
            "prompts_sha256": "3" * 64,
            "model": "claude-sonnet-5-5",
            "claude_code_version": "2.1.5",
            "generated_on": "2026-10-10",
            "specification_commit": "d" * 40,
        })
    )


@dataclass
class _Corpus:
    texts: dict[Language, dict[str, str]]
    sha256: str = CORPUS_SHA
    record: AuditRecord = field(default_factory=audit_record)
    provenance: ProvenanceRecord | None = field(
        default_factory=lambda: ProvenanceRecord(
            generation=_generation(), audit_sha256=OLD_AUDIT_FILE
        )
    )
    audit_file: str = NEW_AUDIT_FILE
    saved: list[ProvenanceRecord] = field(default_factory=list)

    def notes(self, language: Language) -> dict[str, str]:
        return self.texts.get(language, {})

    def fingerprint(self) -> str | None:
        return self.sha256

    def audit_record(self) -> AuditRecord:
        return self.record

    def audit_sha256(self) -> str:
        return self.audit_file

    def provenance_record(self) -> ProvenanceRecord | None:
        return self.provenance

    def save_provenance(self, record: ProvenanceRecord) -> None:
        self.saved.append(record)


@dataclass(frozen=True)
class _Source:
    notes_: tuple[TranslatedNote, ...]
    record: Translation = field(default_factory=translation_record)

    def translation(self) -> Translation:
        return self.record

    def notes(self) -> tuple[TranslatedNote, ...]:
        return self.notes_


def _texts() -> dict[Language, dict[str, str]]:
    return {
        Language.EN: {"n1": "one ___", "n2": "two ___ ___"},
        Language.CS: {"n1": "jedna ___\n", "n2": "dva ___\n\nx\n"},
    }


def _replace(
    texts: dict[Language, dict[str, str]], language: Language, **changes: str | None
) -> dict[Language, dict[str, str]]:
    changed = {**texts[language], **changes}
    return texts | {
        language: {
            note_id: text for note_id, text in changed.items() if text is not None
        }
    }


def _source() -> _Source:
    return _Source((
        TranslatedNote("n1", sha256_text("one ___"), ("jedna ___",)),
        TranslatedNote("n2", sha256_text("two ___ ___"), ("dva ___", "x")),
    ))


def _run(corpus: _Corpus, source: _Source) -> ProvenanceRefused | ProvenanceWritten:
    return TranslationProvenance(corpus, source).run(PINNED, STRUCTURE_LABELS)


def test_success_keeps_generation_and_writes_translation() -> None:
    """Zápis ponechá generování, doplní translation s neshodami a nový audit_sha256."""
    corpus = _Corpus(_texts())

    assert _run(corpus, _source()) == ProvenanceWritten()

    (saved,) = corpus.saved
    assert saved.generation == _generation()
    assert saved.audit_sha256 == NEW_AUDIT_FILE
    assert saved.translation is not None
    assert saved.translation.marker_mismatches == ("n2",)
    assert saved.translation.model == translation_record().model


def test_missing_provenance_is_refused() -> None:
    """Bez provenance generování není co doplňovat."""
    corpus = _Corpus(_texts(), provenance=None)

    assert isinstance(_run(corpus, _source()), ProvenanceRefused)
    assert corpus.saved == []


def test_corpus_not_matching_audit_is_refused() -> None:
    """Korpus změněný po auditu provenance nedostane."""
    corpus = _Corpus(_texts(), sha256="d" * 64)

    assert isinstance(_run(corpus, _source()), ProvenanceRefused)
    assert corpus.saved == []


def test_corpus_without_czech_is_refused() -> None:
    """Bez cs/ není co k čemu vázat."""
    corpus = _Corpus({Language.EN: {"n1": "one ___"}})

    assert isinstance(_run(corpus, _source()), ProvenanceRefused)
    assert corpus.saved == []


@pytest.mark.parametrize(
    "change",
    [
        lambda texts: _replace(texts, Language.EN, n1="changed ___"),
        lambda texts: _replace(texts, Language.CS, n3="tři\n"),
        lambda texts: _replace(texts, Language.CS, n2=None),
        lambda texts: _replace(texts, Language.CS, n1="upraveno ___\n"),
    ],
    ids=["changed-original", "extra-note", "missing-note", "edited-czech"],
)
def test_corpus_not_bound_to_workdir_is_refused(
    change: Callable[[dict[Language, dict[str, str]]], dict[Language, dict[str, str]]],
) -> None:
    """Změněný originál, jiná množina note_id i ručně upravený překlad zápis zablokují."""
    corpus = _Corpus(change(_texts()))

    outcome = _run(corpus, _source())

    assert isinstance(outcome, ProvenanceRefused)
    assert corpus.saved == []


def test_incomplete_translation_is_refused_before_binding() -> None:
    """Prázdný překlad zablokuje zápis, i když sedí s výstupy pracovního adresáře."""
    texts = {Language.EN: {"n1": "one ___"}, Language.CS: {"n1": ""}}
    source = _Source((TranslatedNote("n1", sha256_text("one ___"), ("",)),))
    corpus = _Corpus(texts)

    outcome = _run(corpus, source)

    assert isinstance(outcome, ProvenanceRefused)
    assert "prázdný" in outcome.reason
    assert corpus.saved == []


def test_masked_translation_binds_to_restored_czech() -> None:
    """Masking z provenance překladu řídí obnovu značek při vazbě na cs/."""
    source = _Source(
        (
            TranslatedNote("n1", sha256_text("one ___"), ("jedna [[1]]",)),
            TranslatedNote("n2", sha256_text("two ___ ___"), ("dva [[1]]", "x [[2]]")),
        ),
        translation_record(masking="[[n]]"),
    )
    texts = _replace(_texts(), Language.CS, n2="dva ___\n\nx ___\n")
    corpus = _Corpus(texts)

    assert _run(corpus, source) == ProvenanceWritten()

    (saved,) = corpus.saved
    assert saved.translation is not None
    assert saved.translation.masking == "[[n]]"


def test_old_provenance_without_masking_loads() -> None:
    """Dnešní provenance bez pole masking se načte s None."""
    fields = translation_record().model_dump(mode="json")
    del fields["masking"]

    assert Translation.model_validate_json(json.dumps(fields)).masking is None
