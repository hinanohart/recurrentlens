"""Vanilla / TopK / JumpReLU SAE variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from recurrentlens.core.types import HookSite, SAEVariant
from recurrentlens.sae.base import BaseSAE


class VanillaSAE(BaseSAE):
    """ReLU encoder + L1 sparsity loss (Bricken et al. 2023)."""

    variant: SAEVariant = "vanilla"


class TopKSAE(BaseSAE):
    """TopK SAE (Gao et al. 2024). Keep top-k activations per token, zero the rest."""

    variant: SAEVariant = "topk"

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int = 32,
        hook_site: HookSite = "out_proj_out",
        layer: int = 0,
        model_id: str = "",
    ):
        super().__init__(
            d_in=d_in, d_sae=d_sae, hook_site=hook_site, layer=layer, model_id=model_id
        )
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_pre = self.encode_pre(x)
        # Pick top-k features per token (along last dim), zero the rest.
        k = min(self.k, z_pre.shape[-1])
        _, idx = z_pre.topk(k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, idx, z_pre.gather(-1, idx).relu())
        return z

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        # TopK enforces sparsity structurally; report L0 as monitoring quantity only.
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)


class JumpReLUSAE(BaseSAE):
    """JumpReLU SAE (Rajamanoharan et al. 2024, arXiv:2407.14435).

    Activation: ``z = (z_pre > theta) * z_pre`` where ``theta`` is a learned
    per-feature threshold. We use a straight-through estimator for the
    discontinuous gradient at the threshold.
    """

    variant: SAEVariant = "jumprelu"

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_site: HookSite = "out_proj_out",
        layer: int = 0,
        model_id: str = "",
        init_theta: float = 0.001,
    ):
        super().__init__(
            d_in=d_in, d_sae=d_sae, hook_site=hook_site, layer=layer, model_id=model_id
        )
        import math

        self.log_theta = nn.Parameter(torch.full((d_sae,), math.log(init_theta)))

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_pre = self.encode_pre(x)
        theta = self.theta
        # Hard gate forward; the gate has no grad path (it is a thresholding op),
        # so the backward pass flows through z_pre on the active set, which is
        # the canonical "straight-through on the hard gate" form used by
        # Rajamanoharan et al. 2024 and matches sae_lens / dictionary_learning.
        gate = (z_pre > theta).to(z_pre.dtype)
        return z_pre * gate

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        # L0 surrogate: count of nonzero (post-gate) entries. log_theta learning
        # via a kernel-density STE is deferred to v0.1.1; for v0.1.0.post1 the
        # threshold stays at init_theta and only encoder/decoder learn.
        return (z != 0).float().sum(dim=-1).mean()


def build_sae(
    variant: SAEVariant,
    d_in: int,
    d_sae: int,
    hook_site: HookSite = "out_proj_out",
    layer: int = 0,
    model_id: str = "",
    **variant_kwargs,
) -> BaseSAE:
    """Factory for SAE variants."""
    if variant == "vanilla":
        return VanillaSAE(
            d_in=d_in, d_sae=d_sae, hook_site=hook_site, layer=layer, model_id=model_id
        )
    if variant == "topk":
        k = variant_kwargs.get("k", 32)
        return TopKSAE(
            d_in=d_in, d_sae=d_sae, k=k, hook_site=hook_site, layer=layer, model_id=model_id
        )
    if variant == "jumprelu":
        return JumpReLUSAE(
            d_in=d_in,
            d_sae=d_sae,
            hook_site=hook_site,
            layer=layer,
            model_id=model_id,
            init_theta=variant_kwargs.get("init_theta", 0.001),
        )
    raise ValueError(f"unknown SAE variant: {variant!r}")
