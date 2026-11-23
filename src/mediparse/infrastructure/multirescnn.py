"""MultiResCNN s maskovanými konvolucemi a pozorností po labelech."""

from __future__ import annotations

import math
import os
from itertools import batched
from typing import TYPE_CHECKING, Final, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.word2vec import trained_vectors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.representation import Device

# Li a Yu 2020, Table 2: šest měřítek a jeden reziduální blok.
KERNEL_SIZES: Final = (3, 5, 9, 15, 19, 25)


class _ResidualBranch(nn.Module):
    def __init__(self, embedding_dim: int, filter_maps: int, kernel: int) -> None:
        super().__init__()
        self.initial = nn.Conv1d(
            embedding_dim, embedding_dim, kernel, padding=kernel // 2
        )
        self.first = nn.Conv1d(embedding_dim, filter_maps, kernel, padding=kernel // 2)
        self.second = nn.Conv1d(filter_maps, filter_maps, kernel, padding=kernel // 2)
        self.shortcut = nn.Conv1d(embedding_dim, filter_maps, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.initial(x)) * mask
        hidden = torch.tanh(self.first(x)) * mask
        hidden = self.second(hidden) * mask
        return torch.tanh(hidden + self.shortcut(x) * mask) * mask


class _MultiResCnn(nn.Module):
    def __init__(
        self, embeddings: torch.Tensor, labels: int, filter_maps: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=False, padding_idx=0
        )
        self.dropout = nn.Dropout(dropout)
        self.branches = nn.ModuleList([
            _ResidualBranch(embeddings.shape[1], filter_maps, kernel)
            for kernel in KERNEL_SIZES
        ])
        channels = len(KERNEL_SIZES) * filter_maps
        self.attention = nn.Linear(channels, labels, bias=False)
        self.output = nn.Linear(channels, labels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0)
        x = self.dropout(self.embedding(tokens)).transpose(1, 2)
        hidden = torch.cat(
            [branch(x, mask[:, None, :]) for branch in self.branches], dim=1
        ).transpose(1, 2)
        attention = (
            self
            .attention(hidden)
            .masked_fill(~mask[:, :, None], -torch.inf)
            .softmax(dim=1)
        )
        pooled = attention.transpose(1, 2) @ hidden
        return (pooled * self.output.weight).sum(dim=2) + self.output.bias


class MultiResCnnClassifier(BaseEstimator):
    """Učí všechna skóre jedním fitem a obnoví nejnižší stop BCE."""

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        random_state: int = 0,
        device: Device = "cpu",
        epochs: int = 20,
        batch_size: int = 16,
        max_words: int = 4000,
        warmup_steps: int = 2000,
        embedding_dim: int = 100,
        min_count: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        filter_maps: int = 50,
    ) -> None:
        """Uloží hyperparametry a technické podmínky tréninku."""
        self.random_state = random_state
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_words = max_words
        self.warmup_steps = warmup_steps
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.filter_maps = filter_maps

    def fit(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        stop_documents: Sequence[str],
        stop_labels: np.ndarray,
    ) -> Self:
        """Naučí síť a slovník jen na fit části; stop část vybírá epochu.

        Returns:
            Tentýž estimator s obnovenými nejlepšími vahami.

        Raises:
            InvalidInputError: Prázdný slovník nebo nedostatek paměti zařízení.
        """
        # Sonda 6: determinismus float32 dává bitově stejné výsledky bez podstatné ceny.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(mode=True)
        torch.manual_seed(self.random_state)
        try:
            return self._fit(documents, labels, stop_documents, stop_labels)
        except torch.OutOfMemoryError:
            msg = "MultiResCNN nemá dostatek paměti zařízení."
            raise InvalidInputError(msg) from None

    def _fit(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        stop_documents: Sequence[str],
        stop_labels: np.ndarray,
    ) -> Self:
        # Skops musí předcházet transformers: opačné pořadí selhává při hledání důvěryhodných typů.
        from transformers import get_linear_schedule_with_warmup  # ruff: ignore[import-outside-top-level]

        vectors = trained_vectors(
            {
                "vector_size": self.embedding_dim,
                "min_count": self.min_count,
                "workers": 1,
                "seed": self.random_state,
            },
            [document.split() for document in documents],
        )
        self.vocabulary_ = {
            word: index + 2 for index, word in enumerate(vectors.index_to_key)
        }
        embeddings = torch.from_numpy(
            np.vstack([
                np.zeros((2, self.embedding_dim), dtype=np.float32),
                vectors.vectors,
            ])
        )
        self.model_ = _MultiResCnn(
            embeddings, labels.shape[1], self.filter_maps, self.dropout
        ).to(self.device)
        train = self._encode(documents)
        target = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        schedule = get_linear_schedule_with_warmup(
            optimizer,
            self.warmup_steps,
            self.epochs * math.ceil(len(train) / self.batch_size),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        self.stop_losses_: list[float] = []
        self.best_epoch_ = self.epochs
        best_state = None
        best_loss = math.inf
        stop = self._encode(stop_documents)
        stop_target = torch.as_tensor(
            stop_labels, dtype=torch.float32, device=self.device
        )
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            order = torch.randperm(len(train), generator=generator).tolist()
            batches = []
            for chunk in batched(order, self.batch_size * 50, strict=False):
                batches.extend(
                    batched(
                        sorted(chunk, key=lambda i: len(train[i])),
                        self.batch_size,
                        strict=False,
                    )
                )
            for index in torch.randperm(len(batches), generator=generator).tolist():
                indices = list(batches[index])
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.model_(self._pad(train, indices)), target[indices]
                )
                loss.backward()
                optimizer.step()
                schedule.step()
            self.model_.eval()
            with torch.inference_mode():
                loss = float(
                    nn.functional.binary_cross_entropy_with_logits(
                        self._logits(stop), stop_target
                    ).item()
                )
            self.stop_losses_.append(loss)
            if math.isfinite(loss) and loss < best_loss:
                best_loss = loss
                self.best_epoch_ = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.model_.state_dict().items()
                }
        self.converged_ = best_state is not None
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _encode(self, documents: Sequence[str]) -> list[torch.Tensor]:
        encoded = []
        for document in documents:
            indices = [
                self.vocabulary_.get(word, 1)
                for word in document.split()[: self.max_words]
            ]
            encoded.append(torch.tensor(indices or [1], dtype=torch.long))
        return encoded

    def _pad(
        self, documents: list[torch.Tensor], indices: Sequence[int]
    ) -> torch.Tensor:
        return nn.utils.rnn.pad_sequence(
            [documents[i] for i in indices], batch_first=True
        ).to(self.device)

    def _logits(self, documents: list[torch.Tensor]) -> torch.Tensor:
        scores = torch.empty(
            (len(documents), self.model_.output.out_features), device=self.device
        )
        for indices in batched(
            sorted(range(len(documents)), key=lambda i: len(documents[i])),
            self.batch_size,
            strict=False,
        ):
            scores[list(indices)] = self.model_(self._pad(documents, indices))
        return scores

    def decision_function(self, documents: Sequence[str]) -> np.ndarray:
        """Vrátí logity ve vstupním pořadí a float64.

        Returns:
            Konečná skóre zpráv v pořadí sloupců labelů fitu.
        """
        self.model_.eval()
        with torch.inference_mode():
            scores = (
                self._logits(self._encode(documents)).cpu().numpy().astype(np.float64)
            )
        # Max-F1 odmítá nekonečná skóre; divergovaný pokus nesmí zastavit ladění.
        return np.nan_to_num(scores, nan=0.0)
