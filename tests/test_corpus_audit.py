"""Jádro auditu syntetického korpusu: normalizace, sdílené 13-gramy, otisk a pravidla brány."""

import pytest
from pydantic import ValidationError

from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NgramIndex,
    ReferenceNote,
    audit_record_violations,
    fingerprint,
    reference_mismatch,
    scan,
    tokenize,
)
from tests.support import (
    CORPUS_SHA,
    DISCHARGE,
    NOTE,
    PINNED,
    RADIOLOGY,
    SENTENCE,
    audit_record,
)

ADMISSIONS = {"name": "admissions.csv.gz", "sha256": "f" * 64, "rows": 30}
DISCHARGE_FILE, RADIOLOGY_FILE = PINNED.items()


def _index(text: str = f"Summary: {SENTENCE}.") -> NgramIndex:
    return NgramIndex({NOTE: text})


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
        audit_record(shared_ngrams=1)


def test_record_requires_timezone() -> None:
    """Čas auditu bez časové zóny by nešel jednoznačně porovnat."""
    with pytest.raises(ValidationError):
        audit_record(created_at="2026-09-28T00:00:00")


def test_gate_requires_record_for_existing_corpus() -> None:
    """Korpus bez záznamu auditu neprojde."""
    assert audit_record_violations(CORPUS_SHA, None, PINNED)


def test_gate_rejects_corpus_changed_after_audit() -> None:
    """Změna korpusu po auditu změní otisk a brána ji zachytí."""
    assert audit_record_violations("d" * 64, audit_record(), PINNED)


def test_gate_rejects_record_from_other_method() -> None:
    """Záznam z jiné délky n-gramu nebo normalizace už korpus neatestuje."""
    assert audit_record_violations(
        CORPUS_SHA, audit_record(ngram_size=NGRAM_SIZE - 1), PINNED
    )
    assert audit_record_violations(
        CORPUS_SHA, audit_record(normalization="other"), PINNED
    )


def test_gate_accepts_matching_record() -> None:
    """Shodný otisk, metoda i reference bránou projdou."""
    assert audit_record_violations(CORPUS_SHA, audit_record(), PINNED) == ()


@pytest.mark.parametrize(
    "reference",
    [
        pytest.param([DISCHARGE], id="bez-radiology"),
        pytest.param([DISCHARGE, RADIOLOGY, ADMISSIONS], id="soubor-navic"),
        pytest.param([DISCHARGE | {"sha256": "d" * 64}, RADIOLOGY], id="jiny-otisk"),
        pytest.param([DISCHARGE, DISCHARGE, RADIOLOGY], id="discharge-dvakrat"),
    ],
)
def test_gate_rejects_reference_other_than_pinned(
    reference: list[dict[str, object]],
) -> None:
    """Audit, který neběžel přesně proti discharge a radiology s připnutými otisky, korpus neatestuje."""
    assert audit_record_violations(
        CORPUS_SHA, audit_record(reference=reference), PINNED
    )


def test_gate_rejects_record_when_config_names_no_reference() -> None:
    """Config bez tabulek MIMIC-IV-Note nemá s čím srovnávat a korpus neatestuje."""
    assert audit_record_violations(CORPUS_SHA, audit_record(), {})


@pytest.mark.parametrize(
    ("files", "mismatch"),
    [
        pytest.param([DISCHARGE_FILE, RADIOLOGY_FILE], (), id="presna-shoda"),
        pytest.param([DISCHARGE_FILE], ("radiology.csv.gz",), id="bez-radiology"),
        pytest.param(
            [DISCHARGE_FILE, RADIOLOGY_FILE, ("admissions.csv.gz", "f" * 64)],
            ("admissions.csv.gz",),
            id="soubor-navic",
        ),
        pytest.param(
            [("discharge.csv.gz", "d" * 64), RADIOLOGY_FILE],
            ("discharge.csv.gz",),
            id="jiny-otisk",
        ),
        pytest.param(
            [DISCHARGE_FILE, DISCHARGE_FILE, RADIOLOGY_FILE],
            ("discharge.csv.gz",),
            id="discharge-dvakrat",
        ),
    ],
)
def test_reference_mismatch_names_differing_files(
    files: list[tuple[str, str]], mismatch: tuple[str, ...]
) -> None:
    """Pravidlo shody jmenuje chybějící, přebývající, zdvojené i jinak otištěné soubory."""
    assert reference_mismatch(files, PINNED) == mismatch
