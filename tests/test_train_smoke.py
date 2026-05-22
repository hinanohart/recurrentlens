"""Phase 5 train loop smoke test on synthetic activations."""

import numpy as np
import torch

from recurrentlens.sae.cache import ActivationCache
from recurrentlens.sae.train import SAETrainConfig, train_sae_from_cache


def test_train_decreases_loss_on_synthetic_data():
    rng = np.random.default_rng(0)
    d_in = 16
    n = 4096

    # Synthetic activations: low-rank signal + noise. SAE should learn the signal.
    W_true = rng.standard_normal((4, d_in)).astype(np.float32)
    codes = rng.gamma(0.5, 1.0, size=(n, 4)).astype(np.float32) * (rng.random((n, 4)) < 0.4)
    acts = codes @ W_true + 0.05 * rng.standard_normal((n, d_in)).astype(np.float32)

    cache = ActivationCache(capacity=n, d_in=d_in, backend="ram")
    cache.append(acts)

    config = SAETrainConfig(
        d_in=d_in,
        d_sae=32,
        variant="topk",
        k=4,
        n_warmup_steps=20,
        log_every=20,
        renormalize_every=20,
        batch_size=256,
        lr=1e-3,
    )
    torch.manual_seed(0)
    train_sae_from_cache(cache, config=config, n_steps=200)

    losses = [row["loss"] for row in config.log_metrics]
    assert len(losses) >= 5
    # Loss should decrease appreciably from the early steps to the late steps.
    early = sum(losses[:2]) / 2
    late = sum(losses[-2:]) / 2
    assert late < early * 0.85, f"loss didn't decrease enough: early={early:.4f} late={late:.4f}"


def test_train_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    d_in = 8
    cache = ActivationCache(capacity=512, d_in=d_in, backend="ram")
    cache.append(rng.standard_normal((512, d_in)).astype(np.float32))

    config = SAETrainConfig(
        d_in=d_in, d_sae=16, variant="vanilla", n_warmup_steps=5, batch_size=64, lr=1e-3
    )
    sae = train_sae_from_cache(cache, config=config, n_steps=30)

    path = tmp_path / "trained.safetensors"
    sae.save(str(path))
    from recurrentlens.sae.base import BaseSAE

    loaded = BaseSAE.load(str(path))
    assert loaded.variant == "vanilla"
    assert loaded.d_in == d_in
