"""Dolaďování transformeru se ztrátou až po spojení oken zprávy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import batched, chain
from time import perf_counter
from typing import TYPE_CHECKING, Final, Self

import numpy as np
from sklearn.base import BaseEstimator

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.hf_encoder import load_parts, note_windows

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch

    from mediparse.domain.representation import Aggregation, EncoderSpec

# Kus seskupí podobné délky a zachová náhodnost: při ~6,4 oknech na zprávu
# je to ~100 dávek po 32 oknech pro okno 512 a rozpočet 16 384 tokenů.
BUCKET_NOTES: Final = 512


def batches(
    lengths: Sequence[tuple[int, int]], budget: int, generator: torch.Generator
) -> list[list[int]]:
    """Zabalí celé zprávy do rozpočtu počtu oken krát nejdelší okno.

    Returns:
        Zamíchané dávky; nadlimitní zpráva je v dávce sama.
    """
    import torch  # ruff: ignore[import-outside-top-level]

    order = torch.randperm(len(lengths), generator=generator).tolist()
    result = []
    for chunk in batched(order, BUCKET_NOTES, strict=False):
        batch: list[int] = []
        windows, longest = 0, 0
        for i in sorted(chunk, key=lambda i: lengths[i]):
            count, length = lengths[i]
            if batch and (windows + count) * max(longest, length) > budget:
                result.append(batch)
                batch, windows, longest = [], 0, 0
            batch.append(i)
            windows += count
            longest = max(longest, length)
        if batch:
            result.append(batch)
    return [
        result[i] for i in torch.randperm(len(result), generator=generator).tolist()
    ]


@dataclass(frozen=True)
class EpochReport:
    """Časy a agregáty epochy se skóre celé validace."""

    epoch: int
    seconds: float
    validation_seconds: float
    peak_memory_bytes: int
    stop_loss: float
    validation_scores: np.ndarray


class HfWindowClassifier(BaseEstimator):
    """Transformer s hlavou nad maskovaným průměrem a spojením logitů oken.

    encoder.dtype zde určuje autocast nad vahami float32; load_encoder jím
    naproti tomu určuje dtype samotných vah.
    """

    # Rozhraní estimatoru musí zpřístupnit hyperparametry pro sklearn.set_params.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        encoder: EncoderSpec,
        aggregation: Aggregation = "max",
        freeze_embeddings: bool = True,
        random_state: int = 0,
        epochs: int = 4,
        batch_tokens: int = 16384,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        device: str = "cpu",
    ) -> None:
        """Uloží explicitní hyperparametry bez načítání vah."""
        self.encoder = encoder
        self.aggregation = aggregation
        self.freeze_embeddings = freeze_embeddings
        self.random_state = random_state
        self.epochs = epochs
        self.batch_tokens = batch_tokens
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.device = device

    def fit(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        stop_documents: Sequence[str],
        stop_labels: np.ndarray,
    ) -> Self:
        """Naučí síť a obnoví epochu s nejnižší stop BCE.

        Returns:
            Tentýž estimator s nejlepšími vahami encoderu i hlavy.
        """
        for _ in self.train_epochs(documents, labels, stop_documents, stop_labels):
            pass
        return self

    def train_epochs(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        stop_documents: Sequence[str],
        stop_labels: np.ndarray,
        validation_documents: Sequence[str] = (),
    ) -> Iterator[EpochReport]:
        """Vydá měření epoch bez zpřístupnění validačních labelů modelu.

        Yields:
            Report každé epochy před obnovou nejlepších vah.

        Raises:
            InvalidInputError: Zařízení nemá dostatek paměti.
        """
        import torch  # ruff: ignore[import-outside-top-level]

        try:
            yield from self._train_epochs(
                documents, labels, stop_documents, stop_labels, validation_documents
            )
        except torch.OutOfMemoryError:
            msg = "Transformer nemá dostatek paměti zařízení."
            raise InvalidInputError(msg) from None

    def _train_epochs(
        self,
        documents: Sequence[str],
        labels: np.ndarray,
        stop_documents: Sequence[str],
        stop_labels: np.ndarray,
        validation_documents: Sequence[str],
    ) -> Iterator[EpochReport]:
        import torch  # ruff: ignore[import-outside-top-level]
        from transformers import get_linear_schedule_with_warmup  # ruff: ignore[import-outside-top-level]

        # GPU determinismus má pro transformer významnou cenu; seed zůstává v provenance.
        torch.manual_seed(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)
        self._initialize(labels.shape[1])
        train, stop, validation = [
            self._encode(texts)
            for texts in (documents, stop_documents, validation_documents)
        ]
        target = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        stop_target = torch.as_tensor(
            stop_labels, dtype=torch.float32, device=self.device
        )
        epoch_batches = [
            batches(self._lengths(train), self.batch_tokens, generator)
            for _ in range(self.epochs)
        ]
        total = sum(map(len, epoch_batches))
        optimizer = torch.optim.AdamW(
            chain(self.model_.parameters(), self.head_.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        schedule = get_linear_schedule_with_warmup(
            optimizer, int(self.warmup_ratio * total), total
        )
        self.stop_losses_: list[float] = []
        self.best_epoch_ = self.epochs
        self.converged_ = False
        best_loss, best_state = math.inf, None
        for epoch, planned in enumerate(epoch_batches, 1):
            self.model_.train()
            self.head_.train()
            self._synchronize()
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            start = perf_counter()
            for indices in planned:
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    self._forward([train[i] for i in indices]), target[indices]
                )
                loss.backward()
                optimizer.step()
                schedule.step()
            self._synchronize()
            report = self._evaluate(
                epoch, perf_counter() - start, stop, stop_target, validation
            )
            self.stop_losses_.append(report.stop_loss)
            if math.isfinite(report.stop_loss) and report.stop_loss < best_loss:
                best_loss, self.best_epoch_ = report.stop_loss, epoch
                best_state = (
                    self._encoder_state(),
                    {
                        k: v.detach().to("cpu", copy=True)
                        for k, v in self.head_.state_dict().items()
                    },
                )
            yield report
        self.converged_ = best_state is not None
        if best_state is not None:
            # Chybějí jen neměnné zmražené parametry; jejich kopie by zbytečně zabírala RAM.
            self.model_.load_state_dict(best_state[0], strict=False)
            self.head_.load_state_dict(best_state[1])

    def _evaluate(
        self,
        epoch: int,
        seconds: float,
        stop: Sequence[list[list[int]]],
        stop_target: torch.Tensor,
        validation: Sequence[list[list[int]]],
    ) -> EpochReport:
        import torch  # ruff: ignore[import-outside-top-level]

        self.model_.eval()
        self.head_.eval()
        with torch.inference_mode():
            stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                self._logits(stop), stop_target
            ).item()
        self._synchronize()
        start = perf_counter()
        scores = self._scores(validation)
        self._synchronize()
        validation_seconds = perf_counter() - start
        peak = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.startswith("cuda")
            else 0
        )
        return EpochReport(epoch, seconds, validation_seconds, peak, stop_loss, scores)

    def _initialize(self, labels: int) -> None:
        import torch  # ruff: ignore[import-outside-top-level]

        # mDeBERTa SDPA nemá; vhodnou implementaci pozornosti zvolí transformers.
        self.model_, self.tokenizer_ = load_parts(
            self.encoder.model_copy(
                update={"device": "cpu" if self.device == "cpu" else "cuda"}
            ),
            dtype=torch.float32,
        )
        # Dekorátor transformers ztrácí v ty vazbu self metody to.
        self.model_.to(self.device)  # ty: ignore[invalid-argument-type]
        self.model_.train()
        # Se zmraženými embeddingy teče gradient díky výchozímu use_reentrant=False v transformers 5.18.
        self.model_.gradient_checkpointing_enable()
        if self.freeze_embeddings:
            self.model_.get_input_embeddings().weight.requires_grad_(
                requires_grad=False
            )
        self.head_ = torch.nn.Linear(self.model_.config.hidden_size, labels).to(
            self.device
        )

    def _encoder_state(self) -> dict[str, torch.Tensor]:
        frozen = {
            name
            for name, parameter in self.model_.named_parameters()
            if not parameter.requires_grad
        }
        return {
            k: v.detach().to("cpu", copy=True)
            for k, v in self.model_.state_dict().items()
            if k not in frozen
        }

    def _synchronize(self) -> None:
        import torch  # ruff: ignore[import-outside-top-level]

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _encode(self, documents: Sequence[str]) -> list[list[list[int]]]:
        return [
            note_windows(self.tokenizer_, text, self.encoder.window)
            for text in documents
        ]

    @staticmethod
    def _lengths(documents: Sequence[list[list[int]]]) -> list[tuple[int, int]]:
        return [(len(windows), max(map(len, windows))) for windows in documents]

    def _forward(self, documents: Sequence[list[list[int]]]) -> torch.Tensor:
        import torch  # ruff: ignore[import-outside-top-level]

        windows = [window for document in documents for window in document]
        longest = max(map(len, windows))
        ids = torch.full(
            (len(windows), longest),
            self.tokenizer_.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(ids)
        for i, window in enumerate(windows):
            ids[i, : len(window)] = torch.tensor(window, device=self.device)
            mask[i, : len(window)] = 1
        with torch.autocast(
            torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=self.encoder.dtype == "bfloat16",
        ):
            hidden = self.model_(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state.float()
        weights = mask[..., None].float()
        logits = self.head_((hidden * weights).sum(1) / weights.sum(1))
        groups = logits.split([len(document) for document in documents])
        return torch.stack([
            group.amax(0) if self.aggregation == "max" else group.mean(0)
            for group in groups
        ])

    def _logits(self, documents: Sequence[list[list[int]]]) -> torch.Tensor:
        import torch  # ruff: ignore[import-outside-top-level]

        result = torch.empty(
            (len(documents), self.head_.out_features), device=self.device
        )
        for indices in batches(
            self._lengths(documents),
            self.batch_tokens,
            torch.Generator().manual_seed(self.random_state),
        ):
            result[indices] = self._forward([documents[i] for i in indices])
        return result

    def _scores(self, documents: Sequence[list[list[int]]]) -> np.ndarray:
        import torch  # ruff: ignore[import-outside-top-level]

        with torch.inference_mode():
            scores = self._logits(documents).cpu().numpy().astype(np.float64)
        # Max-F1 odmítá nekonečná skóre; divergovaný pokus nesmí zastavit ladění.
        return np.nan_to_num(scores, nan=0.0)

    def decision_function(self, documents: Sequence[str]) -> np.ndarray:
        """Vrátí konečné logity ve float64 a vstupním pořadí.

        Returns:
            Matice zprávy krát labely fitu.

        Raises:
            InvalidInputError: Zařízení nemá dostatek paměti.
        """
        import torch  # ruff: ignore[import-outside-top-level]

        self.model_.eval()
        self.head_.eval()
        try:
            return self._scores(self._encode(documents))
        except torch.OutOfMemoryError:
            msg = "Transformer nemá dostatek paměti zařízení."
            raise InvalidInputError(msg) from None
