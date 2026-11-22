"""Malý encoder a tokenizér v paměti pro ověření oken a poolingu."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.representation import EncoderSpec

SPEC = EncoderSpec(
    name="test",
    text_preprocessing=TextPreprocessing.NONE,
    model="test",
    revision="a" * 40,
    window=8,
    device="cpu",
    dtype="float32",
)
VERSIONS = {"torch": "test", "transformers": "test", "tokenizers": "test"}


if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerFast


def encoder_parts(
    words: Sequence[str] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Sestaví deterministický ModernBERT bez stahování.

    Returns:
        Model v eval režimu a tokenizér s obalem bos/eos.
    """
    # Torch je jen v prostředí test, transformers musí následovat až po skops.
    import torch  # ruff: ignore[import-outside-top-level]
    from tokenizers import Tokenizer  # ruff: ignore[import-outside-top-level]
    from tokenizers.models import WordLevel  # ruff: ignore[import-outside-top-level]
    from tokenizers.pre_tokenizers import WhitespaceSplit  # ruff: ignore[import-outside-top-level]
    from tokenizers.processors import TemplateProcessing  # ruff: ignore[import-outside-top-level]
    from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast  # ruff: ignore[import-outside-top-level]

    vocabulary = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        *(words if words is not None else [f"w{i}" for i in range(41)]),
    ]
    tokenizer = Tokenizer(
        WordLevel(
            dict(zip(vocabulary, range(len(vocabulary)), strict=True)),
            unk_token=vocabulary[1],
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 3)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=vocabulary[0],
        unk_token=vocabulary[1],
        bos_token=vocabulary[2],
        eos_token=vocabulary[3],
    )
    # ModernBertConfig čte tento parametr z kwargs v __post_init__.
    config = ModernBertConfig(
        vocab_size=max(64, len(vocabulary)),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        local_attention=8,
        global_attn_every_n_layers=1,  # ty: ignore[unknown-argument]
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
    )
    torch.manual_seed(0)
    return AutoModel.from_config(config, attn_implementation="sdpa").eval(), fast
