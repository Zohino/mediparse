"""Okna encoderu, pooling všech pozic a načtení připnutého modelu."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.hf_tokenizers import load_tokenizer, special_tokens
from mediparse.infrastructure.torch_device import require_device

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy import ndarray
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from mediparse.domain.representation import EncoderSpec

ENCODER_PACKAGES: Final = ("torch", "transformers", "tokenizers")


def token_windows(
    content: Sequence[int], prefix: Sequence[int], suffix: Sequence[int], window: int
) -> list[list[int]]:
    """Rozdělí obsah beze ztrát a každé okno obalí speciálními tokeny.

    Returns:
        Okna v pořadí obsahu, pro prázdný obsah jedno prázdné obalené okno.

    Raises:
        InvalidInputError: Okno nemá místo pro obsah.
    """
    capacity = window - len(prefix) - len(suffix)
    if capacity <= 0:
        msg = "Okno encoderu musí pojmout obsah vedle speciálních tokenů."
        raise InvalidInputError(msg)
    return [
        [*prefix, *content[start : start + capacity], *suffix]
        for start in range(0, max(1, len(content)), capacity)
    ]


def note_windows(
    tokenizer: PreTrainedTokenizerBase, text: str, window: int
) -> list[list[int]]:
    """Tokenizuje zprávu a rozdělí ji beze ztrát.

    Returns:
        Obalená okna v pořadí zprávy.
    """
    return token_windows(*_token_parts(tokenizer, text), window)


def _token_parts(
    tokenizer: PreTrainedTokenizerBase, text: str
) -> tuple[list[int], list[int], list[int]]:
    encoded = tokenizer(text, return_special_tokens_mask=True)
    ids, mask = encoded["input_ids"], encoded["special_tokens_mask"]
    start, end = 0, len(ids)
    while start < end and mask[start]:
        start += 1
    while end > start and mask[end - 1]:
        end -= 1
    return ids[start:end], ids[:start], ids[end:]


@dataclass(frozen=True)
class HfMeanEncoder:
    """Zmražený encoder s průměrem přes pozice všech oken zprávy."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    window: int
    device: str

    def embed(self, text: str) -> ndarray:
        """Spočítá vektor zprávy ve float32 bez gradientů.

        Returns:
            Vektor šířky hidden_size v původním pořadí dimenzí.

        Raises:
            InvalidInputError: Tokenizace nebo forward selhaly.
        """
        # Torch chybí v default.
        import torch  # ruff: ignore[import-outside-top-level]

        with _encoder_errors():
            total = torch.zeros(
                self.model.config.hidden_size, dtype=torch.float32, device=self.device
            )
            positions = 0
            with torch.inference_mode():
                for window in note_windows(self.tokenizer, text, self.window):
                    hidden = (
                        self
                        .model(input_ids=torch.tensor([window], device=self.device))
                        .last_hidden_state[0]
                        .float()
                    )
                    total += hidden.sum(0)
                    positions += len(window)
            return (total / positions).cpu().numpy()


def load_encoder(spec: EncoderSpec) -> HfMeanEncoder:
    """Načte model i tokenizér z téhož připnutého lokálního snapshotu.

    Returns:
        Encoder v eval režimu na explicitním zařízení.

    Raises:
        InvalidInputError: Chybí torch či CUDA, model nelze načíst nebo má neplatné okno.
    """
    model, tokenizer = load_parts(spec, attn_implementation="sdpa")
    with _encoder_errors():
        # Dekorátor transformers ztrácí v ty vazbu self metody to.
        model.to(spec.device)  # ty: ignore[invalid-argument-type]
        return HfMeanEncoder(model.eval(), tokenizer, spec.window, spec.device)


def pinned_snapshot(model: str, revision: str) -> str:
    """Stáhne jen soubory připnutého modelu a tokenizéru.

    Returns:
        Lokální adresář snapshotu.
    """
    from huggingface_hub import snapshot_download  # ruff: ignore[import-outside-top-level]

    return snapshot_download(
        model,
        revision=revision,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
            "pytorch_model.bin.index.json",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
            "*.model",
            "vocab.*",
            "merges.txt",
        ],
    )


def load_parts(
    spec: EncoderSpec, **model_options: object
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Načte připnutý model s tokenizérem a ověří kapacitu okna.

    Returns:
        Model a tokenizér ze stejného snapshotu.

    Raises:
        InvalidInputError: Zařízení, model nebo okno nelze použít.
    """
    require_device(spec.device)
    # Odložené importy zachovají default bez torch a pořadí skops/transformers.
    import torch  # ruff: ignore[import-outside-top-level]
    from transformers import AutoModel  # ruff: ignore[import-outside-top-level]

    with _encoder_errors():
        model_options.setdefault("dtype", getattr(torch, spec.dtype))
        directory = pinned_snapshot(spec.model, spec.revision)
        tokenizer = load_tokenizer(directory, spec.revision)
        # Lokální adresář zabrání auto-konverzi vah přes Space a nepřipnutý PR.
        model = AutoModel.from_pretrained(directory, **model_options)
        capacity = model.config.max_position_embeddings
        if not special_tokens(tokenizer) < spec.window <= capacity:
            msg = "Okno encoderu musí přesáhnout obal a nepřekročit kapacitu modelu."
            raise InvalidInputError(msg)
        return model, tokenizer


@contextmanager
def _encoder_errors() -> Generator[None]:
    # Torch chybí v default.
    import torch  # ruff: ignore[import-outside-top-level]

    try:
        yield
    # Doménová chyba dědí z ValueError; její vlastní vysvětlení musí zůstat.
    except InvalidInputError:
        raise
    except (OSError, ValueError, torch.OutOfMemoryError) as error:
        msg = f"Encoder nelze načíst nebo použít: {type(error).__name__}."
        raise InvalidInputError(msg) from None
