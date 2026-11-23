"""Okna beze ztráty obsahu a mean pooling podle pozic všech oken."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import hf_encoder
from mediparse.infrastructure.hf_encoder import (
    HfMeanEncoder,
    load_encoder,
    token_windows,
)
from tests.hf_encoder_support import SPEC, encoder_parts


@example(content=[], prefix=[2], suffix=[3], extra=1)
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
        for token in part[len(prefix) : len(part) - len(suffix)]
    ] == content
    assert all(
        len(part) <= window
        and part[: len(prefix)] == prefix
        and part[len(part) - len(suffix) :] == suffix
        for part in windows
    )
    if not content:
        assert windows == [prefix + suffix]


@pytest.mark.parametrize(
    ("count", "bf16"), [(3, False), (13, False), (0, False), (3, True)]
)
def test_pooling_matches_position_mean(count: int, *, bf16: bool) -> None:
    """Rovná se průměru přes masku pozornosti nad dávkou s výplní.

    Výplň se nepočítá, speciální tokeny ano, sčítá se ve float32.
    """
    torch = pytest.importorskip("torch")
    model, tokenizer = encoder_parts()
    if bf16:
        model = model.to(torch.bfloat16)
    text = " ".join(f"w{i}" for i in range(1, count + 1))
    vocabulary = tokenizer.get_vocab()
    ids = [vocabulary[f"w{i}"] for i in range(1, count + 1)]
    chunks = (
        [[2, *ids[:6], 3], [2, *ids[6:12], 3], [2, *ids[12:], 3]]
        if count == 13
        else [[2, *ids, 3]]
    )
    batch = tokenizer.pad({"input_ids": chunks}, return_tensors="pt")
    with torch.inference_mode():
        states = model(**batch).last_hidden_state.float()
    weights = batch["attention_mask"][..., None].float()
    masked_states = states * weights
    expected = (masked_states.sum((0, 1)) / weights.sum()).numpy()
    actual = HfMeanEncoder(model, tokenizer, 8, "cpu").embed(text)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert actual.dtype == np.float32
    if count == 13:
        assert not np.allclose(
            actual, (masked_states.sum(1) / weights.sum(1)).mean(0).numpy()
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
        with pytest.raises(
            InvalidInputError, match="Okno encoderu musí přesáhnout obal"
        ):
            load_encoder(spec)


def test_cuda_unavailable_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nedostupná CUDA nesmí znamenat tichý přechod na CPU."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(InvalidInputError, match="device"):
        load_encoder(SPEC.model_copy(update={"device": "cuda", "dtype": "bfloat16"}))


def test_load_preserves_revision_local_snapshot_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model a tokenizér sdílejí připnutý snapshot a explicitní numerický režim."""
    torch = pytest.importorskip("torch")
    from transformers import AutoModel  # ruff: ignore[import-outside-top-level]

    model, tokenizer = encoder_parts()
    snapshot = Mock(return_value="/synthetic-model")
    model_loader = Mock(return_value=model)
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(AutoModel, "from_pretrained", model_loader)
    monkeypatch.setattr(hf_encoder, "load_tokenizer", tokenizer_loader)
    spec = SPEC.model_copy(update={"window": 64, "dtype": "bfloat16"})
    load_encoder(spec)
    snapshot.assert_called_once()
    assert snapshot.call_args.args == (spec.model,)
    assert snapshot.call_args.kwargs["revision"] == spec.revision
    model_loader.assert_called_once_with(
        "/synthetic-model", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer_loader.assert_called_once_with("/synthetic-model", spec.revision)


def test_window_requires_space_for_content() -> None:
    """Obal bez místa pro jediný obsahový token je neplatné okno i bez torch."""
    with pytest.raises(InvalidInputError, match="pojmout obsah"):
        token_windows([1], [2], [3], 2)


@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
def test_encoder_preserves_programming_errors(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Chyba programu se nesmí vydávat za odmítnutí vstupu encoderu."""
    pytest.importorskip("torch")
    model, tokenizer = encoder_parts()
    error = error_type("syntetická chyba")
    monkeypatch.setattr(model, "forward", Mock(side_effect=error))
    with pytest.raises(error_type) as caught:
        HfMeanEncoder(model, tokenizer, 8, "cpu").embed("w1")
    assert caught.value is error
