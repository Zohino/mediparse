"""Okna beze ztráty obsahu a mean pooling podle pozic všech oken."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.hf_encoder import (
    HfMeanEncoder,
    load_encoder,
    token_windows,
)
from tests.hf_encoder_support import SPEC, encoder_parts


@given(
    content=st.lists(st.integers(), max_size=200),
    prefix=st.lists(st.integers(), max_size=2),
    suffix=st.lists(st.integers(), max_size=2),
    extra=st.integers(1, 16),
)
def test_windows_preserve_content(
    content: list[int], prefix: list[int], suffix: list[int], extra: int
) -> None:
    """Každý obsahový token zůstane právě jednou v pořadí a s obalem."""
    window = len(prefix) + len(suffix) + extra
    windows = token_windows(content, prefix, suffix, window)
    assert [
        token
        for part in windows
        for token in part[len(prefix) : len(part) - len(suffix) if suffix else None]
    ] == content
    assert all(
        len(part) <= window
        and part[: len(prefix)] == prefix
        and (not suffix or part[-len(suffix) :] == suffix)
        for part in windows
    )
    if not content:
        assert windows == [prefix + suffix]


@pytest.mark.parametrize(
    ("count", "bf16"), [(3, False), (13, False), (0, False), (3, True)]
)
def test_pooling_matches_position_mean(count: int, *, bf16: bool) -> None:
    """Váží pozice, včetně speciálních, a sčítá vždy ve float32."""
    torch = pytest.importorskip("torch")
    model, tokenizer = encoder_parts()
    if bf16:
        model = model.to(torch.bfloat16)
    text = " ".join(f"w{i}" for i in range(1, count + 1))
    ids = [tokenizer.convert_tokens_to_ids(f"w{i}") for i in range(1, count + 1)]
    chunks = (
        [[2, *ids[:6], 3], [2, *ids[6:12], 3], [2, *ids[12:], 3]]
        if count == 13
        else [[2, *ids, 3]]
    )
    with torch.inference_mode():
        states = [
            model(input_ids=torch.tensor([part])).last_hidden_state[0].float()
            for part in chunks
        ]
    expected = torch.cat(states).mean(0).numpy()
    actual = HfMeanEncoder(model, tokenizer, 8, "cpu").embed(text)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert actual.dtype == np.float32
    if count == 13:
        assert not np.allclose(
            actual, torch.stack([state.mean(0) for state in states]).mean(0).numpy()
        )


@pytest.mark.parametrize(
    ("window", "accepted"), [(2, False), (65, False), (3, True), (64, True)]
)
def test_load_validates_window_boundaries(
    monkeypatch: pytest.MonkeyPatch, window: int, *, accepted: bool
) -> None:
    """Okno musí pojmout obsah a nepřekročit kapacitu modelu."""
    pytest.importorskip("torch")
    from transformers import AutoModel  # ruff: ignore[import-outside-top-level]

    from mediparse.infrastructure import hf_encoder  # ruff: ignore[import-outside-top-level]

    model, tokenizer = encoder_parts()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_args, **_kwargs: "/synthetic-model",
    )
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(hf_encoder, "load_tokenizer", lambda *_args: tokenizer)
    spec = SPEC.model_copy(update={"window": window})
    if accepted:
        assert load_encoder(spec).window == window
    else:
        with pytest.raises(InvalidInputError, match=r"okn|Okno"):
            load_encoder(spec)


def test_cuda_unavailable_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nedostupná CUDA nesmí znamenat tichý přechod na CPU."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(InvalidInputError, match="device"):
        load_encoder(SPEC.model_copy(update={"device": "cuda", "dtype": "bfloat16"}))
