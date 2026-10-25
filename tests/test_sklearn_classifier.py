"""Adaptéry scikit-learn: trénink, klasifikace a uložení modelu."""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Final

import pytest
import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from mediparse.domain.clinical_text import (
    NUMBER_PLACEHOLDER,
    TextNormalization,
    expand_diagnosis_acronyms,
    normalize_clinical_text,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import strip_diagnostic_sections
from mediparse.infrastructure.sklearn_classifier import (
    TRUSTED_FUNCTIONS,
    LinearSvmTrainer,
    SkopsModelFile,
    tfidf_preprocessor,
    tfidf_vectorizer,
)
from tests.model_support import (
    ACRONYM_VOCABULARY,
    SPELLING_VOCABULARY,
    separable_notes,
)
from tests.section_support import REPOSITORY_LEXICON, mimic_shaped_notes

if TYPE_CHECKING:
    from pathlib import Path

PATIENTS: Final = 12


def _corpus() -> tuple[list[str], list[bool]]:
    notes = separable_notes(PATIENTS)
    return (
        [note.text for note in notes],
        [Diagnosis.DIABETES in note.labels for note in notes],
    )


def test_classify_returns_label_with_signed_score() -> None:
    """Label odpovídá predikci pipeline a skóre má znaménko podle labelu."""
    texts, labels = _corpus()

    classifier = LinearSvmTrainer(regularization=1.0, seed=0, normalization=None).fit(
        texts, labels
    )
    classifications = classifier.classify(texts)

    assert tuple(item.predicted for item in classifications) == tuple(labels)
    assert tuple(item.predicted for item in classifications) == tuple(
        bool(label) for label in classifier.pipeline.predict(texts)
    )
    assert all((item.score > 0) == item.predicted for item in classifications)


def test_model_file_round_trip(tmp_path: Path) -> None:
    """Uložený model se načte a klasifikuje stejně."""
    texts, labels = _corpus()
    classifier = LinearSvmTrainer(regularization=1.0, seed=0, normalization=None).fit(
        texts, labels
    )
    model_file = SkopsModelFile(tmp_path / "out" / "model.skops")

    model_file.save(classifier)
    loaded = model_file.load()

    assert loaded.classify(texts) == classifier.classify(texts)


def test_model_file_missing_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor modelu je neplatný vstup a jmenuje cestu."""
    path = tmp_path / "model.skops"

    with pytest.raises(InvalidInputError, match=r"model\.skops"):
        SkopsModelFile(path).load()


class _Unsafe:
    pass


def test_model_file_corrupt_is_invalid_input(tmp_path: Path) -> None:
    """Poškozený soubor modelu je neplatný vstup a jmenuje cestu."""
    path = tmp_path / "bad.skops"
    path.write_text("garbage", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"bad\.skops"):
        SkopsModelFile(path).load()


def test_model_file_untrusted_type_is_invalid_input(tmp_path: Path) -> None:
    """Typ mimo seznam důvěryhodných je neplatný vstup a jmenuje cestu."""
    path = tmp_path / "unsafe.skops"
    sio.dump(_Unsafe(), path)

    with pytest.raises(InvalidInputError, match=r"unsafe\.skops"):
        SkopsModelFile(path).load()


_NORMALIZATION: Final = TextNormalization(
    acronyms={
        "COPD": "chronic obstructive pulmonary disease",
        "AF": "atrial fibrillation",
    }
)


def _trainer(normalization: TextNormalization | None) -> LinearSvmTrainer:
    return LinearSvmTrainer(regularization=1.0, seed=0, normalization=normalization)


def _acronym_corpus() -> tuple[list[str], list[bool]]:
    texts = [*ACRONYM_VOCABULARY[:1] * 4, *ACRONYM_VOCABULARY[1:] * 4]
    return texts, [True] * 4 + [False] * 4


def _dose_corpus() -> tuple[list[str], list[bool]]:
    texts = [*SPELLING_VOCABULARY[:1] * 4, *SPELLING_VOCABULARY[1:] * 4]
    return texts, [True] * 4 + [False] * 4


def test_preprocessor_lowercases_after_normalization() -> None:
    """Vlastní preprocessor vypíná lowercase vectorizeru, snížení písmen nese funkce."""
    text = "CAST 500mg ___yo"

    assert tfidf_preprocessor(text, {}) == normalize_clinical_text(text).lower()
    assert "cast" in tfidf_preprocessor(text, {})


def test_preprocessor_expands_acronym_next_to_marker_and_lowercases() -> None:
    """Zkratka za značkou se rozvine před snížením písmen."""
    result = tfidf_preprocessor("___COPD", _NORMALIZATION.acronyms)

    assert "chronic obstructive pulmonary disease" in result
    assert "copd" not in result


def test_trainer_with_acronyms_scores_abbreviation_and_full_name_alike() -> None:
    """Zkratka a vypsaný název dostanou se slovníkem stejné skóre, bez něj různé."""
    texts, labels = _acronym_corpus()

    expanded = _trainer(_NORMALIZATION).fit(texts, labels).classify(ACRONYM_VOCABULARY)
    raw = _trainer(None).fit(texts, labels).classify(ACRONYM_VOCABULARY)

    assert expanded[0].score == expanded[1].score
    assert raw[0].score != raw[1].score


def test_model_file_round_trip_keeps_acronyms(tmp_path: Path) -> None:
    """Načtený model skóruje shodně a jeho preprocessor nese slovník jako data."""
    texts, labels = _acronym_corpus()
    classifier = _trainer(_NORMALIZATION).fit(texts, labels)
    model_file = SkopsModelFile(tmp_path / "model.skops")

    model_file.save(classifier)
    loaded = model_file.load()

    assert loaded.classify(ACRONYM_VOCABULARY) == classifier.classify(
        ACRONYM_VOCABULARY
    )
    preprocessor = loaded.pipeline["tfidf"].preprocessor
    assert preprocessor.keywords == {"acronyms": dict(_NORMALIZATION.acronyms)}
    assert type(preprocessor.keywords["acronyms"]) is dict


def test_model_file_partial_around_foreign_function_is_invalid_input(
    tmp_path: Path,
) -> None:
    """Důvěra v ``functools.partial`` nepustí cizí funkci uvnitř."""
    texts, labels = _corpus()
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=functools.partial(_foreign_preprocessor, acronyms={})
            ),
        ),
        ("svm", LinearSVC()),
    ]).fit(texts, labels)
    path = tmp_path / "foreign_partial.skops"
    sio.dump(pipeline, path)

    with pytest.raises(
        InvalidInputError, match=r"foreign_partial\.skops.*_foreign_preprocessor"
    ):
        SkopsModelFile(path).load()


def test_trainer_with_normalization_scores_spellings_alike() -> None:
    """Se zapnutou normalizací dostanou dva zápisy téže dávky a věku stejné skóre."""
    texts, labels = _dose_corpus()

    normalized = (
        _trainer(_NORMALIZATION).fit(texts, labels).classify(SPELLING_VOCABULARY)
    )
    raw = _trainer(normalization=None).fit(texts, labels).classify(SPELLING_VOCABULARY)

    assert normalized[0].score == normalized[1].score
    assert raw[0].score != raw[1].score


def test_model_file_round_trip_with_preprocessor(tmp_path: Path) -> None:
    """Model s preprocessorem se načte a skóre se shodují."""
    texts, labels = _dose_corpus()
    classifier = _trainer(_NORMALIZATION).fit(texts, labels)
    model_file = SkopsModelFile(tmp_path / "model.skops")

    model_file.save(classifier)

    assert model_file.load().classify(SPELLING_VOCABULARY) == classifier.classify(
        SPELLING_VOCABULARY
    )


def _foreign_preprocessor(text: str, acronyms: object = None) -> str:
    del acronyms
    return text.lower()


def test_model_file_foreign_preprocessor_is_invalid_input(tmp_path: Path) -> None:
    """Funkce mimo seznam důvěryhodných jmen se nenačte."""
    texts, labels = _corpus()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=_foreign_preprocessor)),
        ("svm", LinearSVC()),
    ]).fit(texts, labels)
    path = tmp_path / "foreign.skops"
    sio.dump(pipeline, path)

    with pytest.raises(
        InvalidInputError, match=r"foreign\.skops.*_foreign_preprocessor"
    ):
        SkopsModelFile(path).load()


def test_trusted_functions_are_exact_full_names() -> None:
    """Seznam nese přesné plné jméno funkce, přejmenování spadne tady, ne při načtení."""
    assert (
        "functools.partial",
        f"{tfidf_preprocessor.__module__}.{tfidf_preprocessor.__qualname__}",
    ) == TRUSTED_FUNCTIONS


@pytest.mark.parametrize("name", sorted(mimic_shaped_notes()))
def test_analyzer_sees_normalized_stripped_fixture(name: str) -> None:
    """Analyzátor modelu vidí text po stripu a normalizaci tokeny výchozího vzoru."""
    text = mimic_shaped_notes()[name]
    texts, labels = _dose_corpus()
    analyzer = (
        _trainer(_NORMALIZATION).fit(texts, labels).pipeline["tfidf"].build_analyzer()
    )
    stripped = strip_diagnostic_sections(text, REPOSITORY_LEXICON)

    expected = TfidfVectorizer().build_analyzer()(
        expand_diagnosis_acronyms(
            normalize_clinical_text(stripped), _NORMALIZATION.acronyms
        ).lower()
    )

    tokens = analyzer(stripped)
    assert tokens == expected
    assert not any("___" in token for token in tokens)
    assert not any(token.isdigit() for token in tokens)
    if re.search(r"(?<![A-Za-z0-9_])[0-9]", stripped):
        assert NUMBER_PLACEHOLDER in tokens


def test_analyzer_tokens_after_strip_normalization_and_expansion() -> None:
    """Doslovný seznam tokenů po stripu, normalizaci a rozvinutí, bez produkčních funkcí."""
    texts, labels = _dose_corpus()
    analyzer = _trainer(_NORMALIZATION).fit(texts, labels).pipeline["tfidf"]
    text = "Discharge Diagnosis:\nCOPD\n\nHistory of Present Illness:\nPt ___yo with COPD and AF, 500 mg."

    stripped = strip_diagnostic_sections(text, REPOSITORY_LEXICON)

    assert analyzer.build_analyzer()(stripped) == [
        "history",
        "of",
        "present",
        "illness",
        "pt",
        "__age__",
        "with",
        "chronic",
        "obstructive",
        "pulmonary",
        "disease",
        "and",
        "atrial",
        "fibrillation",
        "__num__",
        "mg",
    ]


def test_vectorizer_without_normalization_has_no_preprocessor_and_lowercases() -> None:
    """Ablace nonorm nemá preprocessor, písmena snižuje sám vectorizer."""
    vectorizer = tfidf_vectorizer(None)

    assert vectorizer.preprocessor is None
    assert vectorizer.build_analyzer()("CAST 5mg") == ["cast", "5mg"]


def test_vectorizer_with_normalization_carries_preprocessor_with_dictionary() -> None:
    """Preprocessor je partial se slovníkem zkratek jako holým dictem."""
    vectorizer = tfidf_vectorizer(_NORMALIZATION)

    assert isinstance(vectorizer.preprocessor, functools.partial)
    assert vectorizer.preprocessor.func is tfidf_preprocessor
    assert vectorizer.preprocessor.keywords == {
        "acronyms": dict(_NORMALIZATION.acronyms)
    }
    assert vectorizer.sublinear_tf


def test_trainer_builds_vectorizer_through_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trenér bere vectorizer z továrny: odlišný token_pattern se projeví v modelu."""
    monkeypatch.setattr(
        "mediparse.infrastructure.sklearn_classifier.tfidf_vectorizer",
        lambda _normalization: TfidfVectorizer(
            token_pattern=r"[a-z]{3,}"  # ruff: ignore[hardcoded-password-func-arg]
        ),
    )
    texts, labels = _dose_corpus()

    model = _trainer(None).fit(texts, labels)

    assert model.pipeline["tfidf"].build_analyzer()("ab abc 12 abcd") == ["abc", "abcd"]


def test_vectorizer_variants_differ_only_in_preprocessor() -> None:
    """Obě varianty z jednoho konstruktoru: liší se jen klíčem preprocessor."""
    plain = tfidf_vectorizer(None).get_params()
    prepared = tfidf_vectorizer(_NORMALIZATION).get_params()

    assert plain.pop("preprocessor") is None
    assert prepared.pop("preprocessor") is not None
    assert plain == prepared
