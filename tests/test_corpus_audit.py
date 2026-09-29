"""Jádro auditu syntetického korpusu: normalizace, sdílené 13-gramy, otisk a pravidla brány."""

import json

import pytest
from pydantic import ValidationError

from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditRecord,
    NgramIndex,
    ReferenceNote,
    fingerprint,
    gate_violations,
    scan,
    tokenize,
)

SENTENCE = (
    "the old lighthouse keeper counted seven gulls before the storm "
    "reached the northern harbor wall"
)
NOTE = "en/90000001-DS-1.txt"
CORPUS_SHA = "a" * 64


def _index(text: str = f"Summary: {SENTENCE}.") -> NgramIndex:
    return NgramIndex({NOTE: text})


def _record(**overrides: object) -> AuditRecord:
    fields = {
        "corpus_sha256": CORPUS_SHA,
        "corpus_files": 1,
        "ngram_size": NGRAM_SIZE,
        "normalization": NORMALIZATION,
        "synthetic_ngrams": 3,
        "reference": [{"name": "discharge.csv.gz", "sha256": "b" * 64, "rows": 10}],
        "shared_ngrams": 0,
        "colliding_subjects": 0,
        "tool_commit": "c" * 40,
        "created_at": "2026-09-28T00:00:00+00:00",
    } | overrides
    return AuditRecord.model_validate_json(json.dumps(fields))


def test_tokenize_normalizes_case_width_and_punctuation() -> None:
    """NFKC sjednotí šířku znaků i ligatury, velikost písmen nerozhoduje a interpunkce odděluje."""
    text = (
        "\N{FULLWIDTH LATIN CAPITAL LETTER P}\N{FULLWIDTH LATIN SMALL LETTER T}. "
        "\N{LATIN SMALL LIGATURE FI}nal-DOSE: 5mg"
    )

    assert tokenize(text) == ["pt", "final", "dose", "5mg"]


def test_tokenize_keeps_deid_marker_as_one_token() -> None:
    """Značka ___ je jeden token, jinde podtržítka tokeny oddělují."""
    assert tokenize("Mr. ___ saw dr_smith") == ["mr", "___", "saw", "dr", "smith"]


def test_planted_sentence_is_found_despite_formatting() -> None:
    """Věta z korpusu se najde i s jinými velkými písmeny, zalomením a interpunkcí."""
    reference = (
        "THE OLD LIGHTHOUSE keeper,\ncounted seven gulls; before the storm "
        "reached the northern harbor wall!"
    )

    shared = _index().shared_with(reference)

    assert len(shared) == len(tokenize(SENTENCE)) - NGRAM_SIZE + 1


def test_twelve_shared_tokens_are_not_a_match() -> None:
    """Hranice kritéria: dvanáct společných tokenů za sebou shodou není."""
    twelve = " ".join(tokenize(SENTENCE)[: NGRAM_SIZE - 1])

    assert not _index().shared_with(twelve)


def test_fullwidth_variant_is_found() -> None:
    """NFKC chytí i text zapsaný plnošířkovými znaky."""
    fullwidth = SENTENCE.translate({code: code + 0xFEE0 for code in range(0x21, 0x7F)})

    assert _index().shared_with(fullwidth)


def test_positions_point_only_to_synthetic_notes() -> None:
    """Pozice shody míří do syntetické zprávy a na token, kde sdílený úsek začíná."""
    index = NgramIndex({
        NOTE: f"Three intro words. {SENTENCE}",
        "cs/90000002-DS-1.txt": "nic společného tu není",
    })

    positions = index.positions(index.shared_with(SENTENCE))

    assert {position.note for position in positions} == {NOTE}
    assert min(position.token for position in positions) == len(
        tokenize("Three intro words.")
    )


def test_index_size_counts_distinct_ngrams() -> None:
    """Velikost indexu je počet různých n-gramů syntetické strany."""
    assert len(_index()) == len(tokenize(f"Summary: {SENTENCE}.")) - NGRAM_SIZE + 1


def test_scan_counts_rows_matches_and_subject_collisions() -> None:
    """Průchod referencí spočítá zprávy, sdílené n-gramy i kolize subject_id."""
    rows = [
        ReferenceNote(subject_id="10000032", text="unrelated reference text"),
        ReferenceNote(subject_id="90000001", text=SENTENCE),
    ]

    result = scan(_index(), frozenset({"90000001"}), rows)

    assert result.rows == len(rows)
    assert result.colliding_subjects == {"90000001"}
    assert result.shared


def test_fingerprint_ignores_order_but_not_content_or_names() -> None:
    """Otisk nezávisí na pořadí souborů, ale na jejich obsahu i jménech ano."""
    files = [("en/a.txt", b"one"), ("cs/a.txt", b"two")]
    base = fingerprint(files)

    assert fingerprint(reversed(files)) == base
    assert fingerprint([("en/a.txt", b"one!"), ("cs/a.txt", b"two")]) != base
    assert fingerprint([("en/b.txt", b"one"), ("cs/a.txt", b"two")]) != base


def test_record_with_findings_is_rejected() -> None:
    """Záznam auditu vzniká jen z čistého auditu, se shodami je neplatný."""
    with pytest.raises(ValidationError):
        _record(shared_ngrams=1)


def test_record_requires_timezone() -> None:
    """Čas auditu bez časové zóny by nešel jednoznačně porovnat."""
    with pytest.raises(ValidationError):
        _record(created_at="2026-09-28T00:00:00")


def test_gate_passes_when_there_is_no_corpus() -> None:
    """Dokud korpus neexistuje, brána nemá co kontrolovat."""
    assert gate_violations(None, None) == ()


def test_gate_requires_record_for_existing_corpus() -> None:
    """Korpus bez záznamu auditu neprojde."""
    assert gate_violations(CORPUS_SHA, None)


def test_gate_rejects_corpus_changed_after_audit() -> None:
    """Změna korpusu po auditu změní otisk a brána ji zachytí."""
    assert gate_violations("d" * 64, _record())


def test_gate_rejects_record_from_other_method() -> None:
    """Záznam z jiné délky n-gramu nebo normalizace už korpus neatestuje."""
    assert gate_violations(CORPUS_SHA, _record(ngram_size=NGRAM_SIZE - 1))
    assert gate_violations(CORPUS_SHA, _record(normalization="other"))


def test_gate_accepts_matching_record() -> None:
    """Shodný otisk i metoda bránou projdou."""
    assert gate_violations(CORPUS_SHA, _record()) == ()
