"""Adaptéry scikit-learn: trénink, klasifikace a uložení modelu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from mediparse.domain.clinical_text import normalize_clinical_text
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import strip_diagnostic_sections
from mediparse.infrastructure.sklearn_classifier import (
    TRUSTED_FUNCTIONS,
    LinearSvmTrainer,
    SkopsModelFile,
    tfidf_preprocessor,
)
from tests.model_support import separable_notes
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

    classifier = LinearSvmTrainer(
        regularization=1.0, seed=0, normalize_clinical_text=False
    ).fit(texts, labels)
    classifications = classifier.classify(texts)

    assert tuple(item.predicted for item in classifications) == tuple(labels)
    assert tuple(item.predicted for item in classifications) == tuple(
        bool(label) for label in classifier.pipeline.predict(texts)
    )
    assert all((item.score > 0) == item.predicted for item in classifications)


def test_model_file_round_trip(tmp_path: Path) -> None:
    """Uložený model se načte s trusted=[] a klasifikuje stejně."""
    texts, labels = _corpus()
    classifier = LinearSvmTrainer(
        regularization=1.0, seed=0, normalize_clinical_text=False
    ).fit(texts, labels)
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


_DOSE_PAIR: Final = (
    "metformin 500mg ___yo",
    "metformin 500 mg ___ year old",
)


def _trainer(*, normalize: bool) -> LinearSvmTrainer:
    return LinearSvmTrainer(
        regularization=1.0, seed=0, normalize_clinical_text=normalize
    )


def _dose_corpus() -> tuple[list[str], list[bool]]:
    texts = [*_DOSE_PAIR[:1] * 4, *_DOSE_PAIR[1:] * 4]
    return texts, [True] * 4 + [False] * 4


def test_preprocessor_lowercases_after_normalization() -> None:
    """Vlastní preprocessor vypíná lowercase vectorizeru, snížení písmen nese funkce."""
    text = "CAST 500mg ___yo"

    assert tfidf_preprocessor(text) == normalize_clinical_text(text).lower()
    assert "cast" in tfidf_preprocessor(text)


def test_trainer_with_normalization_scores_spellings_alike() -> None:
    """Se zapnutou normalizací dostanou dva zápisy téže dávky a věku stejné skóre."""
    texts, labels = _dose_corpus()

    normalized = _trainer(normalize=True).fit(texts, labels).classify(_DOSE_PAIR)
    raw = _trainer(normalize=False).fit(texts, labels).classify(_DOSE_PAIR)

    assert normalized[0].score == normalized[1].score
    assert raw[0].score != raw[1].score


def test_model_file_round_trip_with_preprocessor(tmp_path: Path) -> None:
    """Model s preprocessorem se načte a skóre se shodují."""
    texts, labels = _dose_corpus()
    classifier = _trainer(normalize=True).fit(texts, labels)
    model_file = SkopsModelFile(tmp_path / "model.skops")

    model_file.save(classifier)

    assert model_file.load().classify(_DOSE_PAIR) == classifier.classify(_DOSE_PAIR)


def _foreign_preprocessor(text: str) -> str:
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

    with pytest.raises(InvalidInputError, match=r"foreign\.skops"):
        SkopsModelFile(path).load()


def test_trusted_functions_are_exact_full_names() -> None:
    """Seznam nese přesné plné jméno funkce, přejmenování spadne tady, ne při načtení."""
    assert (
        f"{tfidf_preprocessor.__module__}.{tfidf_preprocessor.__qualname__}",
    ) == TRUSTED_FUNCTIONS


@pytest.mark.parametrize("name", sorted(mimic_shaped_notes()))
def test_analyzer_sees_normalized_stripped_fixture(name: str) -> None:
    """Analyzátor modelu vidí text po stripu a normalizaci tokeny výchozího vzoru."""
    text = mimic_shaped_notes()[name]
    texts, labels = _dose_corpus()
    analyzer = (
        _trainer(normalize=True).fit(texts, labels).pipeline["tfidf"].build_analyzer()
    )
    stripped = strip_diagnostic_sections(text, REPOSITORY_LEXICON)

    expected = TfidfVectorizer().build_analyzer()(
        normalize_clinical_text(stripped).lower()
    )

    assert analyzer(stripped) == expected
