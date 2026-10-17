"""Adaptéry scikit-learn: trénink, klasifikace a uložení modelu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import skops.io as sio

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.sklearn_classifier import (
    LinearSvmTrainer,
    SkopsModelFile,
)
from tests.model_support import separable_notes

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

    classifier = LinearSvmTrainer(regularization=1.0, seed=0).fit(texts, labels)
    classifications = classifier.classify(texts)

    assert tuple(item.predicted for item in classifications) == tuple(labels)
    assert tuple(item.predicted for item in classifications) == tuple(
        bool(label) for label in classifier.pipeline.predict(texts)
    )
    assert all((item.score > 0) == item.predicted for item in classifications)


def test_model_file_round_trip(tmp_path: Path) -> None:
    """Uložený model se načte s trusted=[] a klasifikuje stejně."""
    texts, labels = _corpus()
    classifier = LinearSvmTrainer(regularization=1.0, seed=0).fit(texts, labels)
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
