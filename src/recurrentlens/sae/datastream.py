"""Streaming activation extraction from a model + dataset.

Phase 5. Bridges:

    tokens (HF Dataset streaming) -> model forward + forward-hook -> ActivationCache

Designed to run on CPU for smoke tests (small n_tokens) and on a single GPU
for the Colab notebook in Phase 10. ``state-spaces/mamba-130m-hf`` paired
with ``HuggingFaceFW/fineweb-edu`` is the canonical small smoke combo.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from recurrentlens.core.types import HookSite
from recurrentlens.hooks.registry import HookHandle, register_hook
from recurrentlens.sae.cache import ActivationCache


def _stream_dataset_text(dataset: str, split: str = "train") -> Iterator[str]:
    """Yield raw text examples from a HuggingFace streaming dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if text:
            yield text


def tokens_to_activations(
    model: Any,
    hook_site: HookSite,
    layer: int,
    token_iter: Iterator[Any],
    cache: ActivationCache,
    n_tokens: int,
    batch_size: int = 32,
    progress: bool = False,
) -> ActivationCache:
    """Fill ``cache`` with up to ``n_tokens`` activations from the hooked site.

    ``token_iter`` yields tensors of shape (B, T) already on the model's device.
    """
    import torch

    handle: HookHandle | None = register_hook(
        model, site=hook_site, layer=layer, transform=lambda t: t.detach().cpu()
    )

    iterator = token_iter
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(token_iter, desc="extract", total=None)
        except ImportError:
            pass

    try:
        for tokens in iterator:
            with torch.no_grad():
                _ = model.forward(tokens)
            captured = handle.activations
            for act in captured:
                # act shape: (B, T, d_in)
                cache.append(act.reshape(-1, act.shape[-1]).numpy())
            handle.clear()
            if cache.is_full or len(cache) >= n_tokens:
                break
    finally:
        if handle is not None:
            handle.remove()

    return cache


def make_token_iter_from_text(
    tokenizer: Any,
    text_iter: Iterator[str],
    seq_len: int = 1024,
    batch_size: int = 8,
    device: str = "cpu",
) -> Iterator[Any]:
    """Convert a text iterator into a (B, T) integer-token tensor iterator."""
    import torch

    buf: list[int] = []
    while True:
        try:
            text = next(text_iter)
        except StopIteration:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= seq_len * batch_size:
            chunk = buf[: seq_len * batch_size]
            buf = buf[seq_len * batch_size :]
            t = torch.tensor(chunk, dtype=torch.long, device=device).reshape(batch_size, seq_len)
            yield t
