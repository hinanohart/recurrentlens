"""Sparse autoencoder base class.

Phase 4. Concrete variants (vanilla / topk / jumprelu) live in ``variants.py``.

Design notes
------------
- Encoder: linear (``d_in`` → ``d_sae``) with bias.
- Decoder: linear (``d_sae`` → ``d_in``) with optional pre-bias.
- Pre-bias subtracted from inputs before encoding ("centered" form).
- Decoder weights are unit-norm-renormalised after each optimizer step to
  prevent the trivial "scale W_dec, divide W_enc" degenerate solution.

The base class is variant-agnostic; sparsity is applied by ``encode``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from recurrentlens.core.types import HookSite, SAEVariant


class BaseSAE(nn.Module):
    """Generic SAE: encoder, decoder, save/load."""

    variant: SAEVariant = "vanilla"

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_site: HookSite = "out_proj_out",
        layer: int = 0,
        model_id: str = "",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.hook_site = hook_site
        self.layer = layer
        self.model_id = model_id

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Variant-specific sparsity (subclasses override)."""
        return torch.relu(self.encode_pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def renormalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return ((x - x_hat) ** 2).mean()

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """L1 sparsity, scaled by decoder norm. Override in TopK/JumpReLU."""
        return z.abs().sum(dim=-1).mean()

    def loss(
        self,
        x: torch.Tensor,
        l1_coeff: float = 5e-3,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x_hat, z = self.forward(x)
        recon = self.reconstruction_loss(x, x_hat)
        sparsity = self.sparsity_loss(z)
        total = recon + l1_coeff * sparsity
        with torch.no_grad():
            l0 = (z > 0).float().sum(dim=-1).mean().item()
        return total, {"recon_mse": recon.item(), "l1": sparsity.item(), "l0": l0}

    def save(self, path: str) -> None:
        from safetensors.torch import save_file

        meta = {
            "variant": self.variant,
            "d_in": str(self.d_in),
            "d_sae": str(self.d_sae),
            "hook_site": self.hook_site,
            "layer": str(self.layer),
            "model_id": self.model_id,
        }
        save_file(self.state_dict(), path, metadata=meta)

    @classmethod
    def load(cls, path: str) -> BaseSAE:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        from recurrentlens.sae.variants import build_sae

        sae = build_sae(
            variant=meta.get("variant", "vanilla"),  # type: ignore[arg-type]
            d_in=int(meta["d_in"]),
            d_sae=int(meta["d_sae"]),
            hook_site=meta.get("hook_site", "out_proj_out"),  # type: ignore[arg-type]
            layer=int(meta.get("layer", 0)),
            model_id=meta.get("model_id", ""),
        )
        sae.load_state_dict(tensors)
        return sae
