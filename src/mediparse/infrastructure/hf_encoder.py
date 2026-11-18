"""Okna encoderu, pooling všech pozic a načtení připnutého modelu."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.hf_tokenizers import load_tokenizer

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy import ndarray
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from mediparse.domain.screening import EncoderSpec

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
        # Torch chybí v default; transformers musí následovat až po skops.
        import torch  # ruff: ignore[import-outside-top-level]

        with _encoder_errors():
            content, prefix, suffix = _token_parts(self.tokenizer, text)
            total = torch.zeros(
                self.model.config.hidden_size, dtype=torch.float32, device=self.device
            )
            positions = 0
            with torch.inference_mode():
                for window in token_windows(content, prefix, suffix, self.window):
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
        InvalidInputError: Chybí CUDA, model nelze načíst nebo má neplatné okno.
    """
    # Odložené importy zachovají default bez torch a pořadí skops/transformers.
    import torch  # ruff: ignore[import-outside-top-level]
    from huggingface_hub import snapshot_download  # ruff: ignore[import-outside-top-level]
    from transformers import AutoModel  # ruff: ignore[import-outside-top-level]

    if spec.device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA není dostupná; změň device v configu (změní se i klíč cache)."
        raise InvalidInputError(msg)
    with _encoder_errors():
        directory = snapshot_download(
            spec.model,
            revision=spec.revision,
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
        tokenizer = load_tokenizer(directory, spec.revision)
        # Lokální adresář zabrání auto-konverzi vah přes Space a nepřipnutý PR.
        model = AutoModel.from_pretrained(
            directory, dtype=getattr(torch, spec.dtype), attn_implementation="sdpa"
        )
        _, prefix, suffix = _token_parts(tokenizer, "")
        wrapper = len(prefix) + len(suffix)
        if spec.window <= wrapper or spec.window > model.config.max_position_embeddings:
            msg = "Okno encoderu musí přesáhnout obal a nepřekročit kapacitu modelu."
            raise InvalidInputError(msg)
        return HfMeanEncoder(
            model.to(spec.device).eval(), tokenizer, spec.window, spec.device
        )


@contextmanager
def _encoder_errors() -> Generator[None]:
    try:
        yield
    except InvalidInputError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        msg = f"Encoder nelze načíst nebo použít: {type(error).__name__}."
        raise InvalidInputError(msg) from None
