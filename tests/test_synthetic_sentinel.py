"""Kanárek pokrývá buňky, vazby korpusu i těla všech částí zpráv."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.section_headers import split_sections
from mediparse.domain.synthetic_bhc import bhc_rows
from mediparse.domain.synthetic_mimic import corpus_tables, input_notes
from mediparse.domain.synthetic_sentinel import section_tokens, sentinel_tables
from mediparse.entrypoints.eda_synthetic_tables import eda_tables
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon
from mediparse.infrastructure.synthetic_corpus import CorpusDirectory
from tests.support import (
    REPOSITORY_CORPUS,
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_MIMIC_MANIFEST,
    REPOSITORY_PLANS,
    REPOSITORY_SECTION_HEADERS,
)
from tests.synthetic_support import (
    EDA_NOTES,
    EDA_PLANS,
    LEXICON,
    PREFIX,
    TOKEN_PATTERN,
    remove_section_tokens,
    repository_tables,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SHAPES: Final = read_manifest(REPOSITORY_MIMIC_MANIFEST)
TEXT_CHARS: Final = 100
EXPECTED_TARGETS: Final = 357
KEYS: Final = {"subject_id", "hadm_id", "note_id"}


@pytest.fixture(params=["F", "repository"])
def corpus(request: pytest.FixtureRequest) -> dict[str, list[dict[str, str]]]:
    """F i repozitářový korpus prochází stejnými invarianty kanárku.

    Returns:
        Neoznačené tabulky společného jádra.
    """
    if request.param == "repository":
        return repository_tables()
    tables = corpus_tables(
        EDA_PLANS,
        EDA_NOTES,
        load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES),
    )
    tables["mimic-iv-bhc"] = bhc_rows(EDA_PLANS, tables["discharge"], LEXICON)
    return tables


def _sources(
    table: str, corpus: Mapping[str, Sequence[Mapping[str, str]]]
) -> Sequence[Mapping[str, str]]:
    if table in corpus:
        return corpus[table]
    return [{key: row[key] for key in KEYS} for row in corpus["discharge"]]


@pytest.mark.parametrize("table", SHAPES)
def test_missing_cells_are_unique_and_manifest_order_is_preserved(
    corpus: dict[str, list[dict[str, str]]],
    table: str,
) -> None:
    """A5a,e: každá nedodaná buňka nese prefix, každý sloupec pořadí manifestu."""
    rows = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)[table]
    assert all(tuple(row) == SHAPES[table].columns for row in rows)
    assert len(rows) == len(_sources(table, corpus))
    cells = []
    for row, source in zip(rows, _sources(table, corpus), strict=True):
        for column, value in row.items():
            if column not in source and column not in KEYS:
                assert value.startswith(f"{PREFIX}-")
                cells.append(value)
            elif column not in KEYS and column != "text":
                assert value == source[column]
    assert len(cells) == len(set(cells))


def test_identity_links_are_consistent(corpus: dict[str, list[dict[str, str]]]) -> None:
    """A5b: stejný klíč pacienta, přijetí i zprávy odkazuje na stejnou buňku."""
    tables = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)
    patients = {
        old["subject_id"]: new["subject_id"]
        for old, new in zip(corpus["patients"], tables["patients"], strict=True)
    }
    admissions = {
        old["hadm_id"]: new["hadm_id"]
        for old, new in zip(corpus["admissions"], tables["admissions"], strict=True)
    }
    notes = {
        old["note_id"]: new["note_id"]
        for old, new in zip(corpus["discharge"], tables["discharge"], strict=True)
    }
    assert all(
        value.startswith(f"{PREFIX}-")
        for value in (*patients.values(), *admissions.values(), *notes.values())
    )
    for table, rows in tables.items():
        for source, row in zip(_sources(table, corpus), rows, strict=True):
            if "subject_id" in row:
                assert row["subject_id"] == patients[source["subject_id"]]
            if "hadm_id" in row:
                assert row["hadm_id"] == admissions[source["hadm_id"]]
            if table == "mimic-iv-bhc":
                assert row["note_id"] == notes[source["note_id"]]
    assert {row["note_id"] for row in tables["radiology"]}.isdisjoint(notes.values())


def test_discharge_text_is_unchanged_after_canary_line(
    corpus: dict[str, list[dict[str, str]]],
) -> None:
    """A5c: první zpráva nese i vzácnou hlavičku, text za dekorací je zachovaný."""
    tables = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)
    tokens = []
    for index, (original, decorated) in enumerate(
        zip(corpus["discharge"], tables["discharge"], strict=True)
    ):
        head, text = decorated["text"].split("\n", 1)
        cell, token = head.split()
        assert cell.startswith(f"{PREFIX}-")
        assert token.isalpha()
        if index == 0:
            header, text = text.split("\n", 1)
            assert header == f"{token.capitalize()}:"
        else:
            assert f"{token.capitalize()}:" not in text.splitlines()
        assert text == original["text"]
        tokens.append(token)
    assert len(tokens) == len(set(tokens))
    assert all(len(row["text"]) >= TEXT_CHARS for row in tables["radiology"])


def test_decoration_is_deterministic_and_does_not_mutate_corpus() -> None:
    """Dva běhy se shodují a nesdílejí mutace s korpusovou vrstvou."""
    corpus = repository_tables()
    original = deepcopy(corpus)
    assert sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS) == sentinel_tables(
        corpus, SHAPES, PREFIX, TEXT_CHARS
    )
    assert set(sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)) == set(SHAPES)
    assert corpus == original


def test_empty_prefix_is_refused() -> None:
    """Prázdný prefix má vlastní diagnostiku nad jinak platným korpusem."""
    corpus = repository_tables()
    with pytest.raises(
        InvalidInputError, match="Prefix syntetických buněk nesmí být prázdný"
    ):
        sentinel_tables(corpus, SHAPES, "", TEXT_CHARS)


def test_empty_columns_are_refused() -> None:
    """Tabulka bez sloupců je odmítnutá při jinak úplném manifestu."""
    corpus = repository_tables()
    shapes = {**SHAPES, "empty": TableShape(records=0, columns=())}
    with pytest.raises(InvalidInputError, match=r"Tabulka empty nemá žádné sloupce\."):
        sentinel_tables(corpus, shapes, PREFIX, TEXT_CHARS)


def test_corpus_column_outside_manifest_is_refused(
    corpus: dict[str, list[dict[str, str]]],
) -> None:
    """Korpusový sloupec mimo manifest se nesmí tiše zahodit."""
    corpus["patients"][0]["unexpected"] = "synthetic"
    with pytest.raises(InvalidInputError, match=r"patients.*unexpected"):
        sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)


def test_corpus_table_outside_manifest_is_refused(
    corpus: dict[str, list[dict[str, str]]],
) -> None:
    """Korpusová tabulka mimo manifest se nesmí tiše zahodit."""
    shapes = {name: shape for name, shape in SHAPES.items() if name != "patients"}
    with pytest.raises(InvalidInputError, match="patients"):
        sentinel_tables(corpus, shapes, PREFIX, TEXT_CHARS)


@pytest.fixture
def section_corpus() -> dict[str, list[dict[str, str]]]:
    """Složí stejný tok dekorace a targetu jako entrypoint nad korpusem repozitáře.

    Returns:
        Tabulky po obou vrstvách kanárku.
    """
    plans = PlansFile(REPOSITORY_PLANS).load()
    notes = input_notes(
        plans, CorpusDirectory(REPOSITORY_CORPUS).notes(Language.EN), Language.EN
    )
    return eda_tables(
        plans,
        notes,
        load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES),
        load_section_lexicon(REPOSITORY_SECTION_HEADERS),
        SHAPES,
        prefix=PREFIX,
        text_chars=TEXT_CHARS,
    )


def test_section_decoration_preserves_original_text() -> None:
    """I1: odstranění připojených tokenů vrátí přesný text korpusu na disku."""
    corpus = repository_tables()
    original = deepcopy(corpus["discharge"])
    decorated = section_tokens(
        corpus["discharge"], load_section_lexicon(REPOSITORY_SECTION_HEADERS), PREFIX
    )
    for source, row in zip(original, decorated, strict=True):
        assert remove_section_tokens(row["text"]) == source["text"]
        assert {k: v for k, v in row.items() if k != "text"} == {
            k: v for k, v in source.items() if k != "text"
        }
    assert corpus["discharge"] == original
    assert decorated == section_tokens(
        original, load_section_lexicon(REPOSITORY_SECTION_HEADERS), PREFIX
    )


def test_each_body_has_one_token_and_preserves_section_boundaries() -> None:
    """I2: stejné sekce a hlavičky nesou právě jeden token na neprázdné tělo."""
    corpus = repository_tables()
    lexicon = load_section_lexicon(REPOSITORY_SECTION_HEADERS)
    decorated = section_tokens(corpus["discharge"], lexicon, PREFIX)
    for source, row in zip(corpus["discharge"], decorated, strict=True):
        before = split_sections(source["text"], lexicon)
        after = split_sections(row["text"], lexicon)
        assert [(p.section, p.header) for p in after] == [
            (p.section, p.header) for p in before
        ]
        for old, new in zip(before, after, strict=True):
            assert len(TOKEN_PATTERN.findall(new.body)) == int(bool(old.body.strip()))


def test_all_discharge_tokens_are_unique(
    section_corpus: dict[str, list[dict[str, str]]],
) -> None:
    """I3: číslování prvního řádku a částí nekoliduje, každý token má DF jedna."""
    tables = section_corpus
    # Svědek hlavičky opakuje token prvního řádku s velkým písmenem.
    tokens = [
        token
        for row in tables["discharge"]
        for token in TOKEN_PATTERN.findall(row["text"])
    ]
    assert tokens
    assert len(tokens) == len(set(tokens))


def test_every_bhc_target_contains_a_token(
    section_corpus: dict[str, list[dict[str, str]]],
) -> None:
    """I4: všech 357 plánů s BHC dá target se stopou jediné zprávy."""
    tables = section_corpus
    plans = PlansFile(REPOSITORY_PLANS).load()
    expected = sum("hospital_course" in plan.sections for plan in plans)
    assert expected == EXPECTED_TARGETS
    targets = tables["mimic-iv-bhc"]
    assert len(targets) == expected
    assert all(TOKEN_PATTERN.search(row["target"]) for row in targets)
    texts = {row["note_id"]: row["text"] for row in tables["discharge"]}
    assert all(
        set(TOKEN_PATTERN.findall(row["target"]))
        <= set(TOKEN_PATTERN.findall(texts[row["note_id"]]))
        for row in targets
    )


def test_manual_section_tokens() -> None:
    """Dvě ruční zprávy mají tokeny částí od c, nezměněné hlavičky i konce řádků."""
    discharge = [
        {
            "note_id": "first",
            "text": "Name: ___\nChief Complaint:\n\nCough.\nHome.\nBrief Hospital Course:\nImproved.\n",
        },
        {
            "note_id": "second",
            "text": "Intro.\r\nChief Complaint:\r\nPain.\r\nBrief Hospital Course:\r\nRecovered.",
        },
    ]
    assert section_tokens(discharge, LEXICON, "x-y") == [
        {
            "note_id": "first",
            "text": "Name: ___ xyc\nChief Complaint:\n\nCough. xyd\nHome.\nBrief Hospital Course:\nImproved. xye\n",
        },
        {
            "note_id": "second",
            "text": "Intro. xyf\r\nChief Complaint:\r\nPain. xyg\r\nBrief Hospital Course:\r\nRecovered. xyh",
        },
    ]


def test_empty_bodies_and_text_are_preserved() -> None:
    """Prázdná těla nemají kam vložit token a nespotřebují jeho číslo."""
    discharge = [
        {"text": "Chief Complaint:\n \nBrief Hospital Course:\nBody."},
        {"text": ""},
    ]
    assert section_tokens(discharge, LEXICON, "x-y") == [
        {"text": "Chief Complaint:\n \nBrief Hospital Course:\nBody. xyc"},
        {"text": ""},
    ]


def test_section_tokens_refuse_empty_prefix() -> None:
    """Ani prázdná tabulka nesmí přijmout prázdný prefix kanárku."""
    with pytest.raises(
        InvalidInputError, match="Prefix syntetických buněk nesmí být prázdný"
    ):
        section_tokens([], LEXICON, "")
