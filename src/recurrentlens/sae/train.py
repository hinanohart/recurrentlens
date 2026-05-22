"""SAE training loop.

Phase 5. Minimal but correct: AdamW, cosine LR schedule, periodic decoder
renormalization, optional intermediate checkpoint.

This is intentionally lightweight. For paper-grade training, users should run
the included Colab notebook on a T4+ (see notebooks/03_train_mamba130m_sae.ipynb).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from recurrentlens.core.types import CacheBackend, HookSite, SAEVariant
from recurrentlens.sae.base import BaseSAE
from recurrentlens.sae.cache import ActivationCache
from recurrentlens.sae.variants import build_sae


@dataclass
class SAETrainConfig:
    d_in: int
    d_sae: int = 16384
    variant: SAEVariant = "topk"
    hook_site: HookSite = "out_proj_out"
    layer: int = 0
    model_id: str = ""
    n_tokens: int = 5_000_000
    batch_size: int = 4096
    lr: float = 3e-4
    l1_coeff: float = 5e-3
    n_warmup_steps: int = 200
    renormalize_every: int = 100
    log_every: int = 50
    k: int = 32
    init_theta: float = 0.001
    cache_backend: CacheBackend = "mmap"
    cache_path: str | None = None
    log_metrics: list[dict] = field(default_factory=list)


def _cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_sae_from_cache(
    cache: ActivationCache,
    config: SAETrainConfig,
    n_steps: int = 1000,
    device: str = "cpu",
) -> BaseSAE:
    """Train an SAE from a filled ActivationCache. CPU-runnable smoke."""
    import numpy as np
    import torch

    sae = build_sae(
        variant=config.variant,
        d_in=config.d_in,
        d_sae=config.d_sae,
        hook_site=config.hook_site,
        layer=config.layer,
        model_id=config.model_id,
        k=config.k,
        init_theta=config.init_theta,
    ).to(device)

    opt = torch.optim.AdamW(sae.parameters(), lr=config.lr, betas=(0.9, 0.999))
    rng = np.random.default_rng(0)

    for step in range(1, n_steps + 1):
        batch = cache.sample(config.batch_size, rng=rng)
        x = torch.as_tensor(batch, device=device, dtype=torch.float32)
        loss, metrics = sae.loss(x, l1_coeff=config.l1_coeff)

        lr = _cosine_lr(step, n_steps, config.n_warmup_steps, config.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % config.renormalize_every == 0:
            sae.renormalize_decoder()

        if step % config.log_every == 0 or step == n_steps:
            row = {"step": step, "lr": lr, "loss": float(loss.item()), **metrics}
            config.log_metrics.append(row)

    return sae


def train_sae_full(
    model: Any,
    hook_site: HookSite = "out_proj_out",
    layer: int = 0,
    variant: SAEVariant = "topk",
    d_sae: int = 16384,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    n_tokens: int = 5_000_000,
    batch_size: int = 4096,
    cache_backend: CacheBackend = "mmap",
    cache_path: str | None = None,
    n_steps: int = 1000,
    save_to: str | None = None,
    **variant_kwargs,
) -> BaseSAE:
    """End-to-end: stream a dataset, extract activations, train an SAE.

    For local CPU smoke, prefer ``train_sae_from_cache`` with synthetic data.
    The full pipeline is shipped here for the Colab notebook path.
    """
    from recurrentlens.sae.datastream import (
        _stream_dataset_text,
        make_token_iter_from_text,
        tokens_to_activations,
    )

    text_iter = _stream_dataset_text(dataset)
    token_iter = make_token_iter_from_text(
        model.tokenizer, text_iter, seq_len=1024, batch_size=8, device=model.device
    )
    cache = ActivationCache(
        capacity=n_tokens,
        d_in=model.d_model,
        backend=cache_backend,
        path=cache_path,
    )
    tokens_to_activations(
        model=model,
        hook_site=hook_site,
        layer=layer,
        token_iter=token_iter,
        cache=cache,
        n_tokens=n_tokens,
        batch_size=batch_size,
    )

    config = SAETrainConfig(
        d_in=model.d_model,
        d_sae=d_sae,
        variant=variant,
        hook_site=hook_site,
        layer=layer,
        model_id=getattr(model, "model_id", ""),
        n_tokens=n_tokens,
        batch_size=batch_size,
        cache_backend=cache_backend,
        cache_path=cache_path,
        **{k: v for k, v in variant_kwargs.items() if k in {"k", "init_theta"}},
    )
    sae = train_sae_from_cache(
        cache, config=config, n_steps=n_steps, device=getattr(model, "device", "cpu")
    )
    if save_to is not None:
        sae.save(save_to)
    return sae
