"""Normalizace klinického textu: věk, značka a čísla s jednotkami na zástupné tokeny."""

from __future__ import annotations

import re
import string
from typing import Final

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer

from mediparse.domain.clinical_text import (
    AGE_PLACEHOLDER,
    DEID_PLACEHOLDER,
    NUMBER_PLACEHOLDER,
    TextNormalization,
    expand_diagnosis_acronyms,
    find_diagnosis_acronyms,
    find_diagnosis_variants,
    normalize_clinical_text,
    rewrite_diagnosis_variants,
)
from mediparse.domain.note_text import deid_markers
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

_ANALYZER: Final = TfidfVectorizer().build_analyzer()
_UNREPLACED_NUMBER: Final = re.compile(r"(?<![A-Za-z0-9_])(?<![0-9][.,])[0-9]")
_TEXT: Final = st.text(
    alphabet=string.ascii_letters + string.digits + "_ -.,/%\n" + "é٣", max_size=40
)


def _tokens(text: str) -> list[str]:
    return _ANALYZER(normalize_clinical_text(text))


@pytest.mark.parametrize(
    "form",
    [
        "___ year old",
        "___ years old",
        "___-year-old",
        "___ yearsold",
        "___yo",
        "___ y/o",
        "___ y.o.",
        "___ YO",
    ],
)
def test_every_age_form_from_eda_becomes_age_placeholder(form: str) -> None:
    """Všechny tvary věku naměřené v MIMIC dají týž zástupný token."""
    assert normalize_clinical_text(f"a {form} woman").split() == [
        "a",
        AGE_PLACEHOLDER,
        "woman",
    ]


@pytest.mark.parametrize("text", ["___ yogurt", "___ y/om"])
def test_age_suffix_continuing_into_letters_is_not_age(text: str) -> None:
    """Přípona věku, za níž pokračují písmena, je jiné slovo."""
    assert AGE_PLACEHOLDER not in normalize_clinical_text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr.___", f"Mr. {DEID_PLACEHOLDER} "),
        ("___'s", f" {DEID_PLACEHOLDER} 's"),
        ("at ___pm", f"at  {DEID_PLACEHOLDER} pm"),
        ("a ___, b", f"a  {DEID_PLACEHOLDER} , b"),
    ],
)
def test_other_marker_becomes_deid_placeholder_with_spaces(
    text: str, expected: str
) -> None:
    """Ostatní značka dostane zástupný token oddělený mezerou od přilepeného okolí."""
    assert normalize_clinical_text(text) == expected


@pytest.mark.parametrize("text", ["____", "____ year old", "____yo", "_2", "__"])
def test_longer_underscore_runs_and_underscore_numbers_stay(text: str) -> None:
    """Delší řady podtržítek nejsou značka a číslo za podtržítkem není samostatné."""
    assert normalize_clinical_text(text) == text


def test_dose_spelling_does_not_change_tokens() -> None:
    """Dávka bez mezery a s mezerou dá po normalizaci stejné tokeny."""
    assert _tokens("metformin 500mg") == _tokens("metformin 500 mg")
    assert _tokens("metformin 500mg") == ["metformin", NUMBER_PLACEHOLDER, "mg"]


@pytest.mark.parametrize(
    ("text", "numbers"),
    [("1.8", 1), ("98.6", 1), ("1,000", 1), ("126/74", 2), ("2nd", 1), ("5.4%", 1)],
)
def test_numbers_become_one_placeholder_each(text: str, numbers: int) -> None:
    """Desetinné číslo a číslo s tisícovým oddělovačem je jedno, řadová přípona se pohltí."""
    assert normalize_clinical_text(text).split().count(NUMBER_PLACEHOLDER) == numbers
    assert not _UNREPLACED_NUMBER.search(normalize_clinical_text(text))


def test_percent_is_separated_from_number() -> None:
    """Procento přilepené k číslu se oddělí mezerou."""
    assert normalize_clinical_text("5.4%").split() == [NUMBER_PLACEHOLDER, "%"]


@pytest.mark.parametrize(
    "text", ["DM2", "HbA1c", "B12", "T2DM", "q6h", "x2", "K3.5", "Cr1.8", "x2.5"]
)
def test_number_inside_word_stays(text: str) -> None:
    """Číslo uvnitř slova a zkratky s číslem se nemění, včetně velikosti písmen."""
    assert normalize_clinical_text(text) == text


def test_empty_text_stays_empty() -> None:
    """Prázdný text zůstane prázdný."""
    assert not normalize_clinical_text("")


@given(_TEXT)
def test_normalization_is_idempotent(text: str) -> None:
    """Druhý průchod už nic nemění."""
    once = normalize_clinical_text(text)
    assert normalize_clinical_text(once) == once


@given(_TEXT)
def test_no_standalone_marker_remains(text: str) -> None:
    """Po normalizaci nezbude samostatná značka ___."""
    assert deid_markers(normalize_clinical_text(text)) == 0


@given(_TEXT)
def test_no_replaceable_number_remains(text: str) -> None:
    """Každá řada ASCII číslic stojí hned za písmenem, číslicí nebo podtržítkem."""
    assert not _UNREPLACED_NUMBER.search(normalize_clinical_text(text))


@given(st.text(alphabet=string.ascii_letters + " .,\n-", max_size=40))
def test_text_without_underscore_and_digits_stays(text: str) -> None:
    """Text bez podtržítka a číslic se nezmění."""
    assert normalize_clinical_text(text) == text


@pytest.mark.parametrize("text", ["1st", "2nd", "3rd", "4th", "2ND"])
def test_ordinal_suffix_is_absorbed_in_any_case(text: str) -> None:
    """Řadová přípona se pohltí bez ohledu na velikost písmen."""
    assert normalize_clinical_text(text).split() == [NUMBER_PLACEHOLDER]


def test_suffix_continuing_into_word_is_not_ordinal() -> None:
    """Přípona, za kterou pokračují písmena, patří slovu."""
    assert normalize_clinical_text("3stools").split() == [NUMBER_PLACEHOLDER, "stools"]


def test_suffix_followed_by_digit_stays_idempotent() -> None:
    """Regrese: 1st2nd se po prvním průchodu už nemění."""
    once = normalize_clinical_text("1st2nd")
    assert normalize_clinical_text(once) == once


@pytest.mark.parametrize("text", ["\u0663", "\uff11\uff12", "a\u0663"])
def test_non_ascii_digits_stay(text: str) -> None:
    """Číslice mimo ASCII nejsou čísla, pravidlo používá jen ASCII číslice."""
    assert normalize_clinical_text(text) == text


def test_ascii_digit_control_for_non_ascii_case() -> None:
    """Kontrola k předchozímu testu: ASCII číslice se nahradí."""
    assert normalize_clinical_text("3").split() == [NUMBER_PLACEHOLDER]


def test_age_attached_to_word_is_separated_by_space() -> None:
    """Věk přilepený k slovu se od něj oddělí mezerou."""
    assert normalize_clinical_text("Pt___yo").split() == ["Pt", AGE_PLACEHOLDER]


_ACRONYMS: Final = {
    "MI": "myocardial infarction",
    "AF": "atrial fibrillation",
    "DM": "diabetes",
    "DM2": "diabetes",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("History of MI.", "History of myocardial infarction."),
        ("mi and Mi stay", "mi and Mi stay"),
        ("AF on ECG", "atrial fibrillation on ECG"),
        ("AFib and af stay", "AFib and af stay"),
        ("DM2 noted", "diabetes noted"),
        ("DM noted", "diabetes noted"),
        ("T2DM and HbA1c stay", "T2DM and HbA1c stay"),
        (
            "COPD/CKD",
            "chronic obstructive pulmonary disease/chronic kidney disease",
        ),
        ("non-AF", "non-atrial fibrillation"),
        ("AF. Next", "atrial fibrillation. Next"),
        ("(COPD)", "(chronic obstructive pulmonary disease)"),
        ("CHF stays", "CHF stays"),
    ],
)
def test_acronyms_are_expanded_case_sensitively_on_word_boundaries(
    text: str, expected: str
) -> None:
    """Zkratka se rozvine jen jako celé slovo a ve velikosti písmen slovníku."""
    assert expand_diagnosis_acronyms(text, _ACRONYMS) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AFé", "atrial fibrillation\u00e9"),
        ("éAF", "\u00e9atrial fibrillation"),
        ("_AF", "_AF"),
        ("AF_", "AF_"),
        ("AF9", "AF9"),
    ],
)
def test_acronym_boundary_is_ascii_like_re2(text: str, expected: str) -> None:
    """Hranice slova je ASCII jako v sešitu 8, ne Unicode hranice."""
    assert expand_diagnosis_acronyms(text, _ACRONYMS) == expected


@pytest.mark.parametrize("text", ["AF and MI", "", "a,  b"])
def test_empty_dictionary_keeps_text(text: str) -> None:
    """Bez slovníku se text nemění; prázdná alternace by jinak vložila rozvinutí mezi znaky."""
    assert expand_diagnosis_acronyms(text, {}) == text


def test_acronym_after_normalization_next_to_deid_marker() -> None:
    """Rozvinutí po normalizaci najde zkratku přilepenou ke značce ``___``."""
    expanded = expand_diagnosis_acronyms(normalize_clinical_text("___COPD"), _ACRONYMS)

    assert expanded.split() == ["__deid__", *_ACRONYMS["COPD"].split()]


_WORDS: Final = st.text(
    alphabet=string.ascii_letters + string.digits + "_ -./()\n", max_size=40
)


_TOKENS: Final = st.lists(
    st.sampled_from([
        *_ACRONYMS,
        *_ACRONYMS.values(),
        "AFib",
        "af",
        "_AF",
        "AF9",
        "x",
    ])
).map(" ".join)


@given(_WORDS | _TOKENS)
def test_expansion_is_idempotent(text: str) -> None:
    """Druhé rozvinutí už nic nezmění."""
    once = expand_diagnosis_acronyms(text, _ACRONYMS)

    assert expand_diagnosis_acronyms(once, _ACRONYMS) == once


@given(_WORDS)
def test_text_without_acronym_word_stays(text: str) -> None:
    """Text, v němž zkratka nestojí jako celé slovo, se nezmění."""
    words = re.split(r"[^A-Za-z0-9_]+", text)
    if any(word in _ACRONYMS for word in words):
        return

    assert expand_diagnosis_acronyms(text, _ACRONYMS) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AF, DM2 and MI", ["AF", "DM2", "MI"]),
        ("DM2 DM", ["DM2", "DM"]),
        ("AFib af _AF AF9 T2DM", []),
        ("non-AF (COPD) AF.", ["AF", "COPD", "AF"]),
        ("", []),
    ],
)
def test_acronyms_are_found_in_order_with_same_boundaries(
    text: str, expected: list[str]
) -> None:
    """Hledání sdílí hranice a přednost delší zkratky s rozvinutím."""
    assert find_diagnosis_acronyms(text, _ACRONYMS) == expected


@pytest.mark.parametrize("text", ["AF and MI", "", " ", " . "])
def test_find_with_empty_dictionary_returns_nothing(text: str) -> None:
    """Prázdná alternace by jinak našla prázdný řetězec mezi znaky."""
    assert find_diagnosis_acronyms(text, {}) == []


@given(_TOKENS)
def test_find_count_equals_number_of_replacements(text: str) -> None:
    """Počet nalezených zkratek je počet zarážek po rozvinutí na zarážku."""
    marker = "jedinecnyznak"
    expanded = expand_diagnosis_acronyms(text, dict.fromkeys(_ACRONYMS, marker))

    assert expanded.count(marker) == len(find_diagnosis_acronyms(text, _ACRONYMS))


def test_text_normalization_sorts_acronyms() -> None:
    """Klíče se seřadí, aby byl JSON deterministický."""
    normalization = TextNormalization(
        variants={}, acronyms={"DM2": "diabetes", "AF": "atrial"}
    )

    assert list(normalization.acronyms) == ["AF", "DM2"]
    assert list(
        TextNormalization.model_validate_json(normalization.model_dump_json()).acronyms
    ) == ["AF", "DM2"]


@pytest.mark.parametrize(
    "acronyms",
    [
        {"": "diabetes"},
        {"DM": ""},
        {"AF": "AF disease"},
        {"AF": "atrial fibrillation (AF)"},
        {"DM": "diabetes", "DM2": "DM2 diabetes"},
    ],
)
def test_text_normalization_rejects_invalid_dictionary(
    acronyms: dict[str, str],
) -> None:
    """Prázdná zkratka, prázdný tvar a tvar se zkratkou slovníku jsou chyba."""
    with pytest.raises(ValidationError):
        TextNormalization(variants={}, acronyms=acronyms)


def test_text_normalization_is_strict_and_forbids_extra() -> None:
    """Pole mimo model a jiný typ než dict jsou chyba."""
    with pytest.raises(ValidationError):
        TextNormalization.model_validate({"acronyms": {}, "variants": {}, "other": 1})
    with pytest.raises(ValidationError):
        TextNormalization.model_validate({
            "acronyms": [("AF", "atrial")],
            "variants": {},
        })


@pytest.mark.parametrize(
    "acronyms",
    [{"AF": b"atrial fibrillation"}, {b"AF": "atrial fibrillation"}],
)
def test_text_normalization_rejects_bytes(acronyms: dict[object, object]) -> None:
    """Strict odmítne bytes v klíči i hodnotě, které lax režim převede na řetězec."""
    with pytest.raises(ValidationError):
        TextNormalization.model_validate({"acronyms": acronyms, "variants": {}})


_VARIANTS: Final = (
    "T1DM",
    "T2DM",
    "IDDM",
    "NIDDM",
    "PNA",
    "afib",
    "a-fib",
    "a fib",
    "chronic obstructive lung disease",
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("T2DM and t2dm; NIDDM", ["T2DM", "T2DM", "NIDDM"]),
        ("AFib, A-FIB and a fib", ["afib", "a-fib", "a fib"]),
        ("afibrillation Xafib pna1", []),
        ("Tafib", []),
        ("pna.", ["PNA"]),
        ("___T2DM", ["T2DM"]),
        ("a\nfib", ["a fib"]),
        ("a  fib", ["a fib"]),
        ("chronic obstructive\nlung disease", ["chronic obstructive lung disease"]),
        ("chronic  obstructive lung disease", ["chronic obstructive lung disease"]),
        ("", []),
    ],
)
def test_variants_match_manual_examples(text: str, expected: list[str]) -> None:
    """Ruční příklady chrání hranice, pořadí a podobu variant z configu."""
    assert find_diagnosis_variants(normalize_clinical_text(text), _VARIANTS) == expected


@given(
    st.lists(st.sampled_from(_VARIANTS), max_size=30),
    st.text(alphabet=" .,;/()\n-", min_size=1, max_size=5),
)
def test_variants_preserve_composed_order(variants: list[str], separator: str) -> None:
    """Nepísmenné oddělovače zachovají varianty v pořadí složení."""
    text = normalize_clinical_text(separator.join(variants))
    assert find_diagnosis_variants(text, _VARIANTS) == variants


def test_variants_empty_dictionary_and_ascii_matching() -> None:
    """Prázdný slovník nic nenajde, Unicode podobné znaky se neshodují."""
    assert find_diagnosis_variants("afib", []) == []
    assert find_diagnosis_variants("E\u017fKD ESKD", ["ESKD"]) == ["ESKD"]


_REWRITE_VARIANTS: Final = {
    "T2DM": "diabetes",
    "NIDDM": "diabetes",
    "IDDM": "diabetes",
    "PNA": "pneumonia",
    "afib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "a fib": "atrial fibrillation",
    "kindey": "kidney",
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("T2DM and t2dm", "diabetes and diabetes"),
        ("NIDDM", "diabetes"),
        (
            "A-Fib, AFIB, a\n fib",
            "atrial fibrillation, atrial fibrillation, atrial fibrillation",
        ),
        ("Xafib pna1 afibrillation", "Xafib pna1 afibrillation"),
        ("pna.", "pneumonia."),
        ("acute KINDEY injury", "acute kidney injury"),
    ],
)
def test_rewrite_variants_matches_manual_examples(text: str, expected: str) -> None:
    """Přepis odpovídá ručnímu orákulu pro hranice, velikost a překlep slova."""
    assert rewrite_diagnosis_variants(text, _REWRITE_VARIANTS) == expected


@given(_TEXT)
def test_empty_variants_keep_any_text(text: str) -> None:
    """Prázdný slovník text nemění."""
    assert rewrite_diagnosis_variants(text, {}) == text


@given(_TEXT | st.lists(st.sampled_from(list(_REWRITE_VARIANTS))).map(" ".join))
def test_rewrite_is_idempotent_and_independent_of_dictionary_order(text: str) -> None:
    """Přepis nemění vlastní výstup a nezávisí na pořadí položek."""
    once = rewrite_diagnosis_variants(text, _REWRITE_VARIANTS)
    assert rewrite_diagnosis_variants(once, _REWRITE_VARIANTS) == once
    assert (
        rewrite_diagnosis_variants(
            text, dict(reversed(list(_REWRITE_VARIANTS.items())))
        )
        == once
    )


@given(st.text(alphabet="xyzXYZ012 _.,\n", max_size=80))
def test_text_without_variant_stays(text: str) -> None:
    """Abeceda bez tvarů slovníku se přepisem nemění."""
    assert rewrite_diagnosis_variants(text, _REWRITE_VARIANTS) == text


@pytest.mark.parametrize(
    ("acronyms", "variants"),
    [
        ({}, {"afib": "atrial", "AFIB": "atrial"}),
        ({"DM": "diabetes"}, {"dm": "diabetes"}),
        ({}, {"afib": "afib rhythm"}),
        ({"AF": "atrial fibrillation"}, {"afib": "AF rhythm"}),
        ({"AF": "afib"}, {"afib": "atrial fibrillation"}),
        ({}, {"": "diabetes"}),
        ({}, {"afib": ""}),
    ],
)
def test_normalization_rejects_colliding_or_nonidempotent_variants(
    acronyms: dict[str, str], variants: dict[str, str]
) -> None:
    """Kolize a náhrady obsahující klíče slovníků by změnily další průchod."""
    with pytest.raises(ValidationError):
        TextNormalization(acronyms=acronyms, variants=variants)


def test_find_rejects_case_insensitive_collision() -> None:
    """Regrese S37a: hledání nesmí tiše vybrat poslední shodný klíč."""
    with pytest.raises(ValueError, match="Duplicitní"):
        find_diagnosis_variants("x", ["afib", "AFIB"])


def test_normalization_sorts_variants_in_json() -> None:
    """Slovník variant má deterministické pořadí i po serializaci."""
    normalization = TextNormalization(
        acronyms={}, variants={"kindey": "kidney", "afib": "atrial"}
    )
    assert list(
        TextNormalization.model_validate_json(normalization.model_dump_json()).variants
    ) == ["afib", "kindey"]


@pytest.mark.parametrize("variant", ["DM mellitus", "dm mellitus"])
def test_variant_cannot_contain_acronym_word(variant: str) -> None:
    """Zkratka uvnitř klíče varianty by měnila výsledek podle pořadí operací."""
    with pytest.raises(ValidationError, match="zkrat"):
        TextNormalization(
            acronyms={"DM": "diabetes"}, variants={variant: "sugar disease"}
        )


_REPOSITORY_NORMALIZATION: Final = load_text_normalization(
    REPOSITORY_DIAGNOSIS_CANDIDATES, variants=True
)


@given(
    _TEXT
    | st.lists(
        st.sampled_from([
            *_REPOSITORY_NORMALIZATION.variants,
            *_REPOSITORY_NORMALIZATION.variants.values(),
            "chronic obstuctive lung disease",
        ])
    ).map(" ".join)
)
@example("chronic obstuctive lung disease")
def test_rewrite_repository_dictionary_is_idempotent(text: str) -> None:
    """Idempotence platí i pro celý slovník včetně navazujících oprav překlepů."""
    once = rewrite_diagnosis_variants(text, _REPOSITORY_NORMALIZATION.variants)
    assert rewrite_diagnosis_variants(once, _REPOSITORY_NORMALIZATION.variants) == once


@pytest.mark.parametrize("variant", ["a  fib", " afib", " ", "a\tfib"])
def test_normalization_rejects_noncanonical_variant_whitespace(variant: str) -> None:
    """Klíč s nesloučenými bílými znaky by neodpovídal nalezenému tvaru."""
    with pytest.raises(ValidationError):
        TextNormalization(acronyms={}, variants={variant: "x"})


@pytest.mark.parametrize(
    ("variants", "pairs"),
    [
        (
            {"aa": "bb", "bb cc": "dd", "dd ee": "ff", "ff gg": "hh"},
            [("bb cc", "dd ee"), ("dd ee", "ff gg")],
        ),
        ({"aa bb": "aa"}, [("aa bb", "aa bb")]),
    ],
    ids=["N06", "repeated-replacement"],
)
def test_normalization_rejects_composed_replacement_chains(
    variants: dict[str, str], pairs: list[tuple[str, str]]
) -> None:
    """Řetěz delší než jedna návaznost se odmítne s výčtem jeho dvojic."""
    with pytest.raises(ValidationError) as error:
        TextNormalization(acronyms={}, variants=variants)

    message = error.value.errors()[0]["msg"]
    for pair in pairs:
        assert repr(pair) in message


@pytest.mark.parametrize(
    "variants",
    [
        {"xx": "aa bb", "bb cc": "dd", "dd ee": "ff"},
        {"xx": "aa bb", "cc aa": "dd", "dd ee": "ff"},
        {"xx": "bb", "aa bb cc": "dd", "dd ee": "ff"},
        {"xx": "A-FIB", "prefix a fib suffix": "dd", "dd ee": "ff"},
        {"xx": "A1_B2", "prefix a1_b2 suffix": "dd", "dd ee": "ff"},
    ],
    ids=["suffix-prefix", "prefix-suffix", "inner-words", "hyphen-case", "ascii-word"],
)
def test_normalization_rejects_chains_across_word_overlaps(
    variants: dict[str, str],
) -> None:
    """Skládání bere v úvahu slova uvnitř variant i překryvy konců náhrad."""
    with pytest.raises(ValidationError, match="Řetěz"):
        TextNormalization(acronyms={}, variants=variants)


def test_normalization_allows_one_composed_replacement() -> None:
    """Jediná návaznost bb→cc, aa cc→dd se dokončí dvěma průchody."""
    normalization = TextNormalization(acronyms={}, variants={"bb": "cc", "aa cc": "dd"})

    assert rewrite_diagnosis_variants("aa bb", normalization.variants) == "dd"


@pytest.mark.parametrize(
    "variants",
    [
        {"xx": "aa cc", "aa bb cc": "dd", "dd ee": "ff"},
        {"xx": "aa", "baab cc": "dd", "dd ee": "ff"},
    ],
    ids=["noncontiguous-words", "part-of-word"],
)
def test_disconnected_words_do_not_compose_variants(variants: dict[str, str]) -> None:
    """Rozptýlená slova ani podřetězec slova nezakládají další návaznost."""
    assert TextNormalization(acronyms={}, variants=variants).variants == variants


def test_acronyms_do_not_count_toward_variant_chain() -> None:
    """Zkratka se rozvine před oběma průchody, jejich rozpočet nespotřebuje."""
    normalization = TextNormalization(
        acronyms={"XX": "aa"}, variants={"aa bb": "cc", "cc dd": "ee"}
    )
    expanded = expand_diagnosis_acronyms("XX bb dd", normalization.acronyms)

    assert rewrite_diagnosis_variants(expanded, normalization.variants) == "ee"


_REWRITE_WORDS: Final = st.sampled_from([
    "aa",
    "bb",
    "cc",
    "dd",
    "ee",
    "aa-bb",
    "bb.cc",
])
_REWRITE_FORMS: Final = st.lists(_REWRITE_WORDS, min_size=1, max_size=3).map(" ".join)


@given(
    variants=st.dictionaries(
        _REWRITE_FORMS, _REWRITE_FORMS | st.sampled_from(["bb-", "x ."]), max_size=5
    ),
    text=st.lists(_REWRITE_WORDS, max_size=15).map(" ".join),
)
@example(
    variants={"aa": "bb", "bb cc": "dd", "dd ee": "ff", "ff gg": "hh"},
    text="aa cc ee gg",
)
@example(variants={"aa": "bb-", "-cc": "dd-", "-ee": "ff"}, text="aa-cc-ee")
@example(variants={"aa": "x .", ". y": "zz", "zz w": "q"}, text="aa. y w")
@example(variants={"aa": "bb-", ".cc": "dd-", ".ee": "ff"}, text="aa.cc.ee")
@example(variants={"aa bb": "aa"}, text="aa bb bb bb")
@example(variants={"bb": "cc", "aa cc": "dd"}, text="aa bb")
def test_rewrite_validated_small_dictionaries_is_idempotent(
    variants: dict[str, str], text: str
) -> None:
    """Každý přijatý slovník nad malou abecedou dává idempotentní přepis."""
    try:
        normalization = TextNormalization(acronyms={}, variants=variants)
    except ValidationError:
        assume(condition=False)
    once = rewrite_diagnosis_variants(text, normalization.variants)
    assert rewrite_diagnosis_variants(once, normalization.variants) == once


@pytest.mark.parametrize(
    "variants",
    [
        {"aa": "bb-", "-cc": "dd-", "-ee": "ff"},
        {"aa": "x .", ". y": "zz", "zz w": "q"},
        {"aa": "bb-", ".cc": "dd-", ".ee": "ff"},
        {"aa-": "bb"},
        {"aa.": "bb"},
    ],
)
def test_normalization_rejects_nonword_variant_edges(variants: dict[str, str]) -> None:
    """Neslovní okraj klíče umožní návaznost závislou na sousedním znaku."""
    with pytest.raises(ValidationError, match="okrajový znak"):
        TextNormalization(acronyms={}, variants=variants)


def test_nonword_replacement_edges_preserve_idempotence() -> None:
    """Neslovní konce náhrad jsou povolené, když klíče mají slovní okraje."""
    normalization = TextNormalization(
        acronyms={}, variants={"aa": "bb-", "cc": "dd.", "ee": "ff"}
    )
    once = rewrite_diagnosis_variants("aa cc ee", normalization.variants)

    assert once == "bb- dd. ff"
    assert rewrite_diagnosis_variants(once, normalization.variants) == once


@pytest.mark.parametrize("variant", ["a", "A1", "_aa_", "a-fib", "aa.bb"])
def test_normalization_accepts_word_variant_edges(variant: str) -> None:
    """Jediné písmeno, číslice a podtržítko na okraji i vnitřní interpunkce projdou."""
    assert TextNormalization(
        acronyms={}, variants={variant: "replacement"}
    ).variants == {variant: "replacement"}


def test_variant_search_allows_nonword_edges() -> None:
    """Hledání pro EDA nevyžaduje invariant slovníku pro dvouprůchodový přepis."""
    assert find_diagnosis_variants(".cc and -ee", [".cc", "-ee"]) == [".cc", "-ee"]
