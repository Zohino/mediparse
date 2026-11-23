"""Učení syntetických značek a invarianty multi-label sítě."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

pytest.importorskip("torch")

import torch

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import multirescnn
from mediparse.infrastructure.multirescnn import MultiResCnnClassifier
from mediparse.infrastructure.word2vec import trained_vectors

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.usefixtures("single_thread")


@pytest.fixture
def single_thread() -> Iterator[None]:
    """Malé konvoluce neplatí režii mnoha CPU vláken.

    Yields:
        Jednovláknové výpočty po dobu testu.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def corpus() -> tuple[list[str], np.ndarray]:
    """Sestaví 60 fit, 20 stop a 40 test zpráv s pěti nezávislými značkami.

    Returns:
        Dokumenty a labely dané výhradně přítomností značky.
    """
    rng = np.random.default_rng(7)
    labels = rng.random((120, 5)) < 0.5
    documents = []
    for row in labels:
        words = [f"slovo{i}" for i in rng.integers(0, 20, size=8)]
        words.extend(f"znacka{j}" for j, present in enumerate(row) if present)
        rng.shuffle(words)
        documents.append(" ".join(words))
    return documents, labels


def small_model(**params: object) -> MultiResCnnClassifier:
    """Sestaví levný CPU model pro syntetické zprávy.

    Returns:
        Síť s osmi embeddingovými dimenzemi.
    """
    return MultiResCnnClassifier(
        embedding_dim=8,
        filter_maps=4,
        max_words=50,
        batch_size=8,
        warmup_steps=0,
        min_count=1,
        lr=1e-2,
        epochs=30,
        dropout=0.0,
    ).set_params(**params)


@pytest.fixture
def trained(corpus: tuple[list[str], np.ndarray]) -> MultiResCnnClassifier:
    """Naučí model bez použití čtyřiceti testovacích zpráv.

    Returns:
        Model vybraný podle dvaceti stop zpráv.
    """
    documents, labels = corpus
    return small_model().fit(
        documents[:60], labels[:60], documents[60:80], labels[60:80]
    )


def test_learns_marker_labels(
    trained: MultiResCnnClassifier, corpus: tuple[list[str], np.ndarray]
) -> None:
    """Každý z pěti labelů má na nezávislých zprávách ROC-AUC alespoň 0,9."""
    documents, labels = corpus
    aucs = roc_auc_score(
        labels[80:], trained.decision_function(documents[80:]), average=None
    )
    assert np.all(aucs >= 0.9), aucs
    assert trained.converged_


def test_restores_epoch_with_lowest_stop_loss(
    corpus: tuple[list[str], np.ndarray],
) -> None:
    """Negované stop labely vyberou dřívější epochu a nezávislá BCE potvrdí obnovu."""
    documents, labels = corpus
    model = small_model(dropout=0.3).fit(
        documents[:60], labels[:60], documents[60:80], ~labels[60:80]
    )
    assert len(model.stop_losses_) == model.epochs
    assert model.best_epoch_ < model.epochs
    assert model.best_epoch_ == np.argmin(model.stop_losses_) + 1
    z = model.decision_function(documents[60:80])
    bce = np.mean(
        np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - (~labels[60:80]) * z
    )
    assert bce == pytest.approx(model.stop_losses_[model.best_epoch_ - 1], rel=1e-5)
    assert bce < model.stop_losses_[-1]


def test_scores_ignore_batch_padding(trained: MultiResCnnClassifier) -> None:
    """Výplň delších sousedů nesmí změnit skóre krátké zprávy."""
    short = "slovo1 znacka0"
    np.testing.assert_allclose(
        trained.decision_function([short])[0],
        trained.decision_function([short, "slovo2 " * 40, "slovo3 " * 30])[0],
        atol=1e-5,
    )


def test_truncates_to_max_words(trained: MultiResCnnClassifier) -> None:
    """Tokeny za max_words neovlivní skóre."""
    text = "slovo1 " * 50
    np.testing.assert_array_equal(
        trained.decision_function([text]),
        trained.decision_function([text + "znacka0 " * 10]),
    )


def test_vocabulary_comes_only_from_fit_documents(
    corpus: tuple[list[str], np.ndarray],
) -> None:
    """Slova pouze ze stop a test části se nedostanou do slovníku."""
    documents, labels = corpus
    model = small_model(epochs=1).fit(
        documents[:60],
        labels[:60],
        [text + " odlozene" for text in documents[60:80]],
        labels[60:80],
    )
    model.decision_function(["testovaci"])
    assert "odlozene" not in model.vocabulary_
    assert "testovaci" not in model.vocabulary_


def test_repeated_fit_is_bitwise_identical(
    trained: MultiResCnnClassifier, corpus: tuple[list[str], np.ndarray]
) -> None:
    """Stejný seed obnoví přesně stejné skóre i při opakovaném fitu."""
    documents, labels = corpus
    before = trained.decision_function(documents[80:])
    trained.fit(documents[:60], labels[:60], documents[60:80], labels[60:80])
    np.testing.assert_array_equal(before, trained.decision_function(documents[80:]))


def test_empty_document_has_finite_scores(trained: MultiResCnnClassifier) -> None:
    """Prázdný dokument má konečná skóre všech labelů ve float64."""
    scores = trained.decision_function([""])
    assert scores.shape == (1, 5)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(
        scores, trained.decision_function(["slovomimoslovnik"])
    )


def test_divergence_has_finite_scores(corpus: tuple[list[str], np.ndarray]) -> None:
    """Žádná konečná stop ztráta označí nekonvergenci a bezpečná skóre."""
    documents, labels = corpus
    with patch(
        "torch.nn.functional.binary_cross_entropy_with_logits",
        side_effect=lambda z, _y: z.sum() * float("nan"),
    ):
        model = small_model(epochs=2).fit(
            documents[:60], labels[:60], documents[60:80], labels[60:80]
        )
    assert not model.converged_
    assert np.isfinite(model.decision_function(documents[80:])).all()


def test_scheduler_and_deterministic_mode(corpus: tuple[list[str], np.ndarray]) -> None:
    """Deset zpráv v dávce čtyř po dvě epochy dává šest kroků rozvrhu."""
    from transformers import get_linear_schedule_with_warmup  # ruff: ignore[import-outside-top-level]

    schedules = []

    def record_schedule(
        optimizer: torch.optim.Optimizer, warmup: int, steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        schedule = get_linear_schedule_with_warmup(optimizer, warmup, steps)
        schedules.append(schedule)
        return schedule

    documents, labels = corpus
    with patch(
        "transformers.get_linear_schedule_with_warmup",
        side_effect=record_schedule,
    ) as schedule:
        small_model(epochs=2, batch_size=4, warmup_steps=2).fit(
            documents[:10], labels[:10], documents[60:80], labels[60:80]
        )
    assert schedule.call_args.args[1:] == (2, 6)
    assert schedules[0].last_epoch == 6
    assert schedules[0].optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=0.0)
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("lr", 0.001),
        ("weight_decay", 0.1),
        ("dropout", 0.3),
        ("filter_maps", 6),
        ("embedding_dim", 12),
        ("random_state", 13),
    ],
)
def test_parameters_change_scores(
    corpus: tuple[list[str], np.ndarray], parameter: str, value: float
) -> None:
    """Laděný parametr i technické podmínky skutečně ovlivňují skóre."""
    documents, labels = corpus
    baseline = small_model(epochs=2).fit(
        documents[:60], labels[:60], documents[60:80], labels[60:80]
    )
    changed = small_model(epochs=2, **{parameter: value}).fit(
        documents[:60], labels[:60], documents[60:80], labels[60:80]
    )
    assert not np.array_equal(
        baseline.decision_function(documents[80:]),
        changed.decision_function(documents[80:]),
    )


def test_min_count_selects_vocabulary(corpus: tuple[list[str], np.ndarray]) -> None:
    """Slovo se dvěma výskyty splní min_count 2 a nesplní min_count 3."""
    documents, labels = corpus
    fit = [
        text + " dvakrat" if i < 2 else text for i, text in enumerate(documents[:60])
    ]
    for min_count in (2, 3):
        model = small_model(epochs=1, min_count=min_count).fit(
            fit, labels[:60], documents[60:80], labels[60:80]
        )
        assert ("dvakrat" in model.vocabulary_) is (min_count == 2)


def test_every_optimizer_step_uses_training_mode(
    corpus: tuple[list[str], np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Režim tréninku platí při každém kroku také ve druhé epoše."""
    documents, labels = corpus
    model = small_model(epochs=2, dropout=0.3)
    original_step = torch.optim.AdamW.step
    training = []

    def record_step(optimizer: torch.optim.AdamW) -> None:
        training.append(model.model_.training)
        original_step(optimizer)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    model.fit(documents[:60], labels[:60], documents[60:80], labels[60:80])
    assert len(training) == 16
    assert all(training)


def test_random_state_seeds_vectors_and_batches(
    corpus: tuple[list[str], np.ndarray],
) -> None:
    """Seed řídí word2vec i generátor pořadí dávek, nejen inicializaci vah."""
    documents, labels = corpus
    with (
        patch.object(
            multirescnn, "trained_vectors", wraps=multirescnn.trained_vectors
        ) as vectors,
        patch("torch.randperm", wraps=torch.randperm) as permutations,
    ):
        small_model(epochs=2, random_state=13).fit(
            documents[:60], labels[:60], documents[60:80], labels[60:80]
        )
    assert vectors.call_args.args[0]["seed"] == 13
    assert permutations.call_count == 4
    assert all(
        call.kwargs["generator"].initial_seed() == 13
        for call in permutations.call_args_list
    )


def test_embeddings_start_from_word2vec(corpus: tuple[list[str], np.ndarray]) -> None:
    """Nulová rychlost učení zachová word2vec na řádcích slov a nulové rezervy."""
    documents, labels = corpus
    model = small_model(lr=0.0, epochs=1).fit(
        documents[:60], labels[:60], documents[60:80], labels[60:80]
    )
    vectors = trained_vectors(
        {"vector_size": 8, "min_count": 1, "workers": 1, "seed": 0},
        [text.split() for text in documents[:60]],
    )
    weights = model.model_.embedding.weight.detach().cpu().numpy()
    for word in vectors.index_to_key:
        np.testing.assert_array_equal(weights[model.vocabulary_[word]], vectors[word])
    np.testing.assert_array_equal(weights[:2], np.zeros((2, 8), dtype=np.float32))


def test_out_of_memory_becomes_domain_error(
    corpus: tuple[list[str], np.ndarray],
) -> None:
    """Nedostatek paměti má doménovou chybu s provozním vysvětlením."""
    documents, labels = corpus
    with (
        patch("torch.optim.AdamW", side_effect=torch.OutOfMemoryError("syntetická")),
        pytest.raises(InvalidInputError, match="nemá dostatek paměti zařízení"),
    ):
        small_model(epochs=1).fit(
            documents[:60], labels[:60], documents[60:80], labels[60:80]
        )
