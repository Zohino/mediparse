"""Log-odds, determinismus a úklid tokenů skutečného fastTextu."""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import fasttext
import numpy as np
import pytest

from mediparse.infrastructure.fasttext_classifier import FastTextClassifier


@pytest.fixture
def corpus() -> tuple[np.ndarray, list[bool]]:
    """Připraví korpus ze sondy s oddělitelnými třídami.

    Returns:
        Sto dokumentů a jejich binární labely.
    """
    return (
        np.array(["insulin glucose"] * 50 + ["fracture cast"] * 50, dtype=object),
        [True] * 50 + [False] * 50,
    )


def test_log_odds_sign_follows_classes(
    tmp_path: Path, corpus: tuple[np.ndarray, list[bool]]
) -> None:
    """Znaménko log-odds odpovídá sondě a prázdný dokument má konečné skóre."""
    model = FastTextClassifier(tmp_path / "scratch").fit(*corpus)
    np.testing.assert_array_equal(model.classes_, [False, True])
    scores = model.decision_function(
        np.array(["insulin glucose", "fracture cast", ""], dtype=object)
    )
    assert scores.shape == (3,)
    assert scores[0] > 0 > scores[1]
    assert np.isfinite(scores).all()


def test_repeated_fit_is_bitwise_identical(
    tmp_path: Path, corpus: tuple[np.ndarray, list[bool]]
) -> None:
    """Dva fity se stejným seedem vracejí bitově shodné log-odds."""
    first = FastTextClassifier(tmp_path / "scratch", random_state=3).fit(*corpus)
    second = FastTextClassifier(tmp_path / "scratch", random_state=3).fit(*corpus)
    assert np.array_equal(
        first.decision_function(corpus[0]), second.decision_function(corpus[0])
    )


def test_training_is_single_threaded_with_seed(
    tmp_path: Path,
    corpus: tuple[np.ndarray, list[bool]],
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Knihovna dostane jedno vlákno a seed estimátoru."""
    spy = Mock(wraps=fasttext.train_supervised)
    monkeypatch.setattr(fasttext, "train_supervised", spy)
    FastTextClassifier(tmp_path / "scratch", random_state=5).fit(*corpus)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["thread"] == 1
    assert spy.call_args.kwargs["seed"] == 5
    assert tuple(capfd.readouterr()) == ("", "")


@pytest.mark.parametrize("error", [None, RuntimeError("Syntetické selhání tréninku.")])
def test_scratch_is_empty_after_fit(
    tmp_path: Path,
    corpus: tuple[np.ndarray, list[bool]],
    monkeypatch: pytest.MonkeyPatch,
    error: RuntimeError | None,
) -> None:
    """Tokeny vzniknou jen ve scratch a nezůstanou ani po výjimce tréninku."""
    scratch = tmp_path / "missing" / "scratch"
    spy = Mock(wraps=fasttext.train_supervised, side_effect=error)
    monkeypatch.setattr(fasttext, "train_supervised", spy)
    with (
        pytest.raises(RuntimeError, match=re.escape("Syntetické selhání tréninku."))
        if error is not None
        else nullcontext()
    ):
        FastTextClassifier(scratch).fit(*corpus)
    assert Path(spy.call_args.kwargs["input"]).is_relative_to(scratch)
    assert list(scratch.iterdir()) == []


def test_label_word_in_text_is_not_a_label(
    tmp_path: Path, corpus: tuple[np.ndarray, list[bool]]
) -> None:
    """Výchozí předpona v textu nemůže vytvořit třetí label."""
    documents, labels = corpus
    documents[0] = documents[-1] = "__label__1 clinic"
    model = FastTextClassifier(tmp_path / "scratch").fit(documents, labels)
    assert set(model.model_.labels) == {"label:False", "label:True"}


def test_single_class_is_rejected(tmp_path: Path) -> None:
    """Binární estimator odmítne jedinou tréninkovou třídu."""
    with pytest.raises(ValueError, match="právě dvě třídy"):
        FastTextClassifier(tmp_path / "scratch").fit(np.array(["clinic"]), [True])
