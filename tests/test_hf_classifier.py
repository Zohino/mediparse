"""Učení nad celými zprávami a spojení logitů oken."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

pytest.importorskip("torch")

import torch

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import hf_classifier
from mediparse.infrastructure.hf_classifier import HfWindowClassifier, batches
from tests.hf_encoder_support import SPEC, encoder_parts

if TYPE_CHECKING:
    from mediparse.domain.representation import Aggregation

pytestmark = pytest.mark.usefixtures("torch_runtime")


@given(
    lengths=st.lists(
        st.tuples(st.integers(1, 6), st.integers(3, 8)), min_size=1, max_size=60
    ),
    budget=st.integers(8, 64),
)
def test_batches_cover_each_note_once(
    lengths: list[tuple[int, int]], budget: int
) -> None:
    """Dávky dodrží rozpočet kromě samostatné nadlimitní zprávy."""
    result = batches(lengths, budget, torch.Generator().manual_seed(7))
    assert sorted(i for batch in result for i in batch) == list(range(len(lengths)))
    for batch in result:
        cost = sum(lengths[i][0] for i in batch) * max(lengths[i][1] for i in batch)
        assert cost <= budget or len(batch) == 1
    assert result == batches(lengths, budget, torch.Generator().manual_seed(7))


@pytest.fixture
def corpus() -> tuple[list[str], np.ndarray]:
    """Sestaví 60/20/40 zpráv s nezávislými značkami jen v jednom ze tří oken.

    Returns:
        Syntetické texty a labely nezávislé na testovaném klasifikátoru.
    """
    rng = np.random.default_rng(7)
    labels = rng.random((120, 5)) < 0.5
    documents = []
    for row in labels:
        words = [f"slovo{i}" for i in rng.integers(0, 20, size=18)]
        positions = rng.choice(18, size=5, replace=False)
        for j, present in enumerate(row):
            if present:
                words[positions[j]] = f"znacka{j}"
        documents.append(" ".join(words))
    return documents, labels


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> HfWindowClassifier:
    """Sestaví testovací encoder z paměti bez síťového přístupu.

    Returns:
        Klasifikátor s malým ModernBERT a oknem šesti obsahových tokenů.
    """
    words = [f"slovo{i}" for i in range(20)] + [f"znacka{i}" for i in range(5)]
    monkeypatch.setattr(
        hf_classifier, "load_parts", lambda *_args, **_kwargs: encoder_parts(words)
    )
    return HfWindowClassifier(
        encoder=SPEC, epochs=30, lr=1e-3, batch_tokens=72, freeze_embeddings=False
    )


@pytest.fixture
def trained(
    model: HfWindowClassifier, corpus: tuple[list[str], np.ndarray]
) -> HfWindowClassifier:
    """Naučí model jen na fit a stop zprávách.

    Returns:
        Síť obnovená podle stop BCE.
    """
    documents, labels = corpus
    return model.fit(documents[:60], labels[:60], documents[60:80], labels[60:80])


def test_learns_marker_in_one_window(
    trained: HfWindowClassifier, corpus: tuple[list[str], np.ndarray]
) -> None:
    """Každá diagnóza dosáhne na nezávislých zprávách ROC-AUC nejméně 0,9."""
    documents, labels = corpus
    aucs = roc_auc_score(
        labels[80:], trained.decision_function(documents[80:]), average=None
    )
    assert np.all(aucs >= 0.9), aucs
    assert trained.converged_
    assert len(trained.stop_losses_) == trained.epochs
    assert trained.best_epoch_ == np.argmin(trained.stop_losses_) + 1
    z = trained.decision_function(documents[60:80])
    bce = np.mean(np.logaddexp(0, z) - labels[60:80] * z)
    assert bce == pytest.approx(min(trained.stop_losses_), rel=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("aggregation", ["max", "mean"])
def test_max_aggregation_matches_window_logits(
    model: HfWindowClassifier,
    corpus: tuple[list[str], np.ndarray],
    aggregation: Aggregation,
    dtype: str,
) -> None:
    """Ruční výpočet každého okna odpovídá definici max i mean."""
    model.set_params(
        epochs=1,
        aggregation=aggregation,
        encoder=SPEC.model_copy(update={"dtype": dtype}),
    )
    documents, labels = corpus
    model.fit(documents[:4], labels[:4], documents[60:64], labels[60:64])
    ids = model.tokenizer_(documents[0])["input_ids"][1:-1]
    logits = []
    with torch.inference_mode():
        for start in range(0, 18, 6):
            window = torch.tensor([[2, *ids[start : start + 6], 3]])
            with torch.autocast(
                "cpu", dtype=torch.bfloat16, enabled=dtype == "bfloat16"
            ):
                hidden = model.model_(input_ids=window).last_hidden_state.float()
            logits.append(model.head_(hidden.mean(1))[0].numpy())
    expected = (
        np.max(logits, axis=0) if aggregation == "max" else np.mean(logits, axis=0)
    )
    assert model.head_.weight.dtype == torch.float32
    np.testing.assert_allclose(
        model.decision_function([documents[0]])[0], expected, atol=1e-6, rtol=0
    )


def test_order_invariant(
    model: HfWindowClassifier, corpus: tuple[list[str], np.ndarray]
) -> None:
    """Padding a pořadí sousedů nemění skóre zprávy."""
    documents, labels = corpus
    model.set_params(epochs=1).fit(
        documents[:4], labels[:4], documents[60:64], labels[60:64]
    )
    texts = [documents[0], "slovo1", documents[1] + " slovo2"]
    permutation = [2, 0, 1]
    scores = model.decision_function(texts)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(
        model.decision_function([texts[i] for i in permutation]),
        scores[permutation],
        atol=1e-5,
    )


def test_freeze_embeddings(
    model: HfWindowClassifier,
    corpus: tuple[list[str], np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddingy zůstanou přesně stejné a první vrstva encoderu se učí."""
    parts = encoder_parts(
        [f"slovo{i}" for i in range(20)] + [f"znacka{i}" for i in range(5)]
    )
    monkeypatch.setattr(hf_classifier, "load_parts", lambda *_args, **_kwargs: parts)
    embedding = parts[0].get_input_embeddings().weight.detach().clone()
    layer = [p.detach().clone() for p in parts[0].layers[0].parameters()]
    documents, labels = corpus
    model.set_params(epochs=2, freeze_embeddings=True).fit(
        documents[:4], labels[:4], documents[60:64], labels[60:64]
    )
    assert torch.equal(embedding, model.model_.get_input_embeddings().weight)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            layer, model.model_.layers[0].parameters(), strict=True
        )
    )


def test_train_epochs_reports_validation(
    model: HfWindowClassifier, corpus: tuple[list[str], np.ndarray]
) -> None:
    """Každá epocha nese čas, stop BCE a skóre celé validace."""
    documents, labels = corpus
    model.set_params(epochs=3)
    reports = list(
        model.train_epochs(
            documents[:4], labels[:4], documents[60:64], labels[60:64], documents[80:]
        )
    )
    assert [r.epoch for r in reports] == [1, 2, 3]
    assert all(
        r.validation_scores.shape == (40, 5)
        and r.seconds > 0
        and np.isfinite(r.stop_loss)
        for r in reports
    )


def test_out_of_memory_refused(
    model: HfWindowClassifier,
    corpus: tuple[list[str], np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OOM ve forwardu je doménové odmítnutí."""
    parts = encoder_parts()
    monkeypatch.setattr(parts[0], "forward", Mock(side_effect=torch.OutOfMemoryError))
    monkeypatch.setattr(hf_classifier, "load_parts", lambda *_args, **_kwargs: parts)
    documents, labels = corpus
    with pytest.raises(InvalidInputError):
        model.fit(documents[:4], labels[:4], documents[60:64], labels[60:64])
