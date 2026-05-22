"""SAE quality metrics: reconstruction MSE, L0 sparsity, CE recovery.

Phase 9. Designed to be called with either:
  (a) a precomputed (N, d_in) activations tensor — fast, no model needed
  (b) a model + token batch — slower but computes CE recovery

CE recovery: substitute the SAE's reconstruction in place of the residual
activation at the SAE's hook_site/layer, then measure the cross-entropy of
the model's next-token predictions. Closer to baseline = better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchReport:
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"BenchReport({body})"

    def __getitem__(self, k: str) -> float:
        return self.metrics[k]

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": dict(self.metrics), "notes": list(self.notes)}


def evaluate(
    sae: Any,
    activations: Any | None = None,
    model: Any | None = None,
    token_batch: Any | None = None,
    metrics: list[str] | None = None,
) -> BenchReport:
    """Compute SAE quality metrics.

    Parameters
    ----------
    sae : trained SAE
    activations : optional (N, d_in) tensor for recon_mse + l0 (no model needed)
    model : optional RecurrentLensModelImpl, needed for ce_recovery
    token_batch : optional (B, T) integer tensor, needed for ce_recovery
    metrics : subset of ["reconstruction_mse", "l0", "ce_recovery"]; default all available
    """
    import torch

    requested = set(metrics or ["reconstruction_mse", "l0", "ce_recovery"])
    report = BenchReport()

    if activations is not None:
        x = activations if isinstance(activations, torch.Tensor) else torch.as_tensor(activations)
        x = x.to(sae.W_enc.dtype)
        with torch.no_grad():
            x_hat, z = sae(x)
            if "reconstruction_mse" in requested:
                mse = ((x - x_hat) ** 2).mean().item()
                report.metrics["reconstruction_mse"] = float(mse)
                denom = x.var(dim=0).mean().item() + 1e-9
                report.metrics["fvu"] = float(mse / denom)  # fraction of variance unexplained
            if "l0" in requested:
                l0 = (z > 0).float().sum(dim=-1).mean().item()
                report.metrics["l0"] = float(l0)

    if "ce_recovery" in requested:
        if model is None or token_batch is None:
            report.notes.append("ce_recovery skipped: requires both `model` and `token_batch`")
        else:
            ce_clean, ce_sae = _ce_recovery(model, sae, token_batch)
            report.metrics["ce_clean"] = float(ce_clean)
            report.metrics["ce_sae"] = float(ce_sae)
            # standard "CE recovery": (ce_zero_ablation - ce_sae) / (ce_zero_ablation - ce_clean)
            # we approximate with ce_clean / ce_sae as a quick proxy in this MVP.
            report.metrics["ce_recovery_proxy"] = float(ce_clean / max(1e-9, ce_sae))

    return report


def _ce_recovery(model: Any, sae: Any, token_batch: Any) -> tuple[float, float]:
    """Compute clean and SAE-substituted next-token CE."""
    import torch
    import torch.nn.functional as F

    from recurrentlens.hooks.registry import _resolve_target

    def _ce(logits: Any, labels: Any) -> float:
        # next-token: shift by one
        lp = logits[..., :-1, :].contiguous()
        lb = labels[..., 1:].contiguous()
        return F.cross_entropy(lp.reshape(-1, lp.shape[-1]), lb.reshape(-1)).item()

    with torch.no_grad():
        out_clean = model.forward(token_batch)
        logits_clean = out_clean.logits if hasattr(out_clean, "logits") else out_clean
        ce_clean = _ce(logits_clean, token_batch)

        layer_module = model.get_layer(sae.layer)
        target = _resolve_target(layer_module, sae.hook_site)

        def _sub_hook(_m: Any, _i: Any, output: Any) -> Any:
            primary = output[0] if isinstance(output, tuple) else output
            d_in = primary.shape[-1]
            flat = primary.reshape(-1, d_in).to(sae.W_enc.dtype)
            x_hat, _ = sae(flat)
            x_hat = x_hat.to(primary.dtype).reshape(primary.shape)
            if isinstance(output, tuple):
                return (x_hat, *output[1:])
            return x_hat

        h = target.register_forward_hook(_sub_hook)
        try:
            out_sae = model.forward(token_batch)
            logits_sae = out_sae.logits if hasattr(out_sae, "logits") else out_sae
            ce_sae = _ce(logits_sae, token_batch)
        finally:
            h.remove()

    return ce_clean, ce_sae
