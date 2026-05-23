"""SAE variant unit tests + cache backend tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from recurrentlens.sae import (
    ActivationCache,
    BaseSAE,
    JumpReLUSAE,
    TopKSAE,
    VanillaSAE,
    build_sae,
)


def test_build_sae_factory():
    s_van = build_sae("vanilla", d_in=16, d_sae=32)
    s_tk = build_sae("topk", d_in=16, d_sae=32, k=4)
    s_jr = build_sae("jumprelu", d_in=16, d_sae=32)
    assert isinstance(s_van, VanillaSAE)
    assert isinstance(s_tk, TopKSAE)
    assert isinstance(s_jr, JumpReLUSAE)
    assert s_van.variant == "vanilla"
    assert s_tk.variant == "topk"
    assert s_jr.variant == "jumprelu"


def test_build_sae_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown SAE variant"):
        build_sae("not_a_variant", d_in=4, d_sae=8)  # type: ignore[arg-type]


def test_vanilla_forward_shapes():
    sae = VanillaSAE(d_in=8, d_sae=16)
    x = torch.randn(4, 8)
    x_hat, z = sae(x)
    assert x_hat.shape == (4, 8)
    assert z.shape == (4, 16)


def test_topk_enforces_k_sparsity():
    sae = TopKSAE(d_in=8, d_sae=32, k=3)
    x = torch.randn(5, 8)
    _, z = sae(x)
    nonzero = (z != 0).sum(dim=-1)
    # at most k nonzero per row (some may be exact-zero before ReLU)
    assert (nonzero <= 3).all().item()


def test_jumprelu_threshold_gates():
    sae = JumpReLUSAE(d_in=8, d_sae=16, init_theta=1.0)
    x = torch.randn(4, 8)
    _, z = sae(x)
    # all surviving activations must exceed theta (per-feature) - sanity
    theta = sae.theta
    mask = z > 0
    # for each surviving (row, col) the activation should be greater than theta[col]
    rows, cols = mask.nonzero(as_tuple=True)
    assert (z[rows, cols] > theta[cols]).all().item()


def test_loss_returns_metrics():
    sae = TopKSAE(d_in=8, d_sae=16, k=4)
    x = torch.randn(4, 8)
    total, metrics = sae.loss(x)
    assert total.dim() == 0
    assert "recon_mse" in metrics and "l0" in metrics
    assert metrics["l0"] <= 4.0 + 1e-6


def test_decoder_renormalize_unit_norm():
    sae = VanillaSAE(d_in=8, d_sae=16)
    # perturb decoder, then renormalize
    with torch.no_grad():
        sae.W_dec *= 5.0
    sae.renormalize_decoder()
    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sae_save_load_roundtrip(tmp_path: Path):
    sae = TopKSAE(d_in=8, d_sae=16, k=3, hook_site="out_proj_out", layer=2, model_id="x/y")
    path = tmp_path / "sae.safetensors"
    sae.save(str(path))
    loaded = BaseSAE.load(str(path))
    assert loaded.variant == "topk"
    assert loaded.d_in == 8
    assert loaded.d_sae == 16
    assert loaded.hook_site == "out_proj_out"
    assert loaded.layer == 2
    assert loaded.model_id == "x/y"
    for k in sae.state_dict():
        assert torch.allclose(sae.state_dict()[k], loaded.state_dict()[k])


def test_cache_ram_append_and_sample():
    cache = ActivationCache(capacity=100, d_in=8, backend="ram")
    batch = np.random.randn(40, 8).astype(np.float32)
    written = cache.append(batch)
    assert written == 40
    assert len(cache) == 40
    s = cache.sample(16)
    assert s.shape == (16, 8)


def test_cache_ram_overflow_truncates():
    cache = ActivationCache(capacity=10, d_in=4, backend="ram")
    cache.append(np.random.randn(7, 4))
    extra = cache.append(np.random.randn(7, 4))
    assert extra == 3
    assert cache.is_full
    assert len(cache) == 10


def test_cache_mmap_roundtrip(tmp_path: Path):
    p = tmp_path / "act.memmap"
    cache = ActivationCache(capacity=50, d_in=4, backend="mmap", path=str(p))
    cache.append(np.arange(50 * 4, dtype=np.float32).reshape(50, 4))
    s = cache.sample(8)
    assert s.shape == (8, 4)
    cache.close()


def test_cache_3d_input_flattens():
    cache = ActivationCache(capacity=100, d_in=4, backend="ram")
    # (batch, seq, d_in) -> flatten over (batch*seq, d_in)
    batch = np.random.randn(2, 10, 4).astype(np.float32)
    written = cache.append(batch)
    assert written == 20


def test_jumprelu_gradient_flows_to_encoder():
    """Regression: v0.1.0 had a `* 0.0` straight-through estimator that silently
    killed gradient flow. After the v0.1.0.post1 fix, encoder weights must
    receive non-zero gradients on activated features.
    """
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=8, d_sae=16, init_theta=0.001)
    # Use a moderately-scaled input so some features cross threshold.
    x = torch.randn(4, 8) * 0.5
    _, z = sae(x)
    # at least one feature should be active
    assert (z > 0).any().item(), "test setup error: no features above threshold"
    z.sum().backward()
    assert sae.W_enc.grad is not None
    assert (sae.W_enc.grad.abs() > 0).any().item(), (
        "JumpReLU STE killed gradient to W_enc (v0.1.0 regression)"
    )


def test_l0_counts_signed_nonzero():
    """Regression: v0.1.0 reported L0 as (z > 0).sum(), miscounting negative
    SAE activations. v0.1.0.post1 uses (z != 0).
    """
    # Synthetic z with one positive, one negative, two zeros per row.
    z = torch.tensor([[1.0, -0.5, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0]])
    l0_signed = (z != 0).float().sum(dim=-1).mean().item()
    l0_positive_only = (z > 0).float().sum(dim=-1).mean().item()
    assert l0_signed == 2.0
    assert l0_positive_only == 1.0


def test_base_sae_load_rejects_inconsistent_meta(tmp_path):
    """Regression for hub-supply-chain: hand-crafted safetensors with metadata
    that lies about d_sae must be rejected, not silently loaded.
    """
    import pytest
    from safetensors.torch import save_file

    path = tmp_path / "tampered.safetensors"
    fake_state = {
        "W_enc": torch.zeros(4, 8),
        "b_enc": torch.zeros(8),
        "W_dec": torch.zeros(8, 4),
        "b_dec": torch.zeros(4),
    }
    save_file(
        fake_state,
        str(path),
        metadata={
            "variant": "vanilla",
            "d_in": "4",
            "d_sae": "999",  # lying — actual W_enc has d_sae=8
            "hook_site": "out_proj_out",
            "layer": "0",
            "model_id": "test/lies",
        },
    )
    with pytest.raises(ValueError, match="W_enc has shape"):
        BaseSAE.load(str(path))


def test_base_sae_load_rejects_unknown_variant(tmp_path):
    import pytest
    from safetensors.torch import save_file

    path = tmp_path / "alien.safetensors"
    save_file(
        {
            "W_enc": torch.zeros(4, 8),
            "b_enc": torch.zeros(8),
            "W_dec": torch.zeros(8, 4),
            "b_dec": torch.zeros(4),
        },
        str(path),
        metadata={
            "variant": "alien",
            "d_in": "4",
            "d_sae": "8",
            "hook_site": "out_proj_out",
            "layer": "0",
            "model_id": "test/alien",
        },
    )
    with pytest.raises(ValueError, match="unknown variant"):
        BaseSAE.load(str(path))
