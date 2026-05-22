"""Feature explorer: render top-activating examples for an SAE feature as HTML.

Phase 8 MVP. Intentionally minimal — a single HTML table with the top-k
activating token strings + activation magnitudes + decoder-weight summary.
Richer interactivity (cross-feature highlights, n-way diff) lives in v0.1.x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>recurrentlens — feature {feature_id} ({model_id})</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
h1 {{ font-size: 1.4em; }}
.meta {{ color: #555; font-size: 0.9em; margin-bottom: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 0.4em 0.8em; border-bottom: 1px solid #eee; vertical-align: top; }}
th {{ background: #fafafa; font-weight: 600; }}
.act {{ font-variant-numeric: tabular-nums; color: #444; }}
.tok {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background: #f5f5f5; padding: 0.05em 0.35em; border-radius: 3px; }}
.bar {{ display: inline-block; height: 0.6em; background: linear-gradient(to right, #d0e8ff, #2a7ae0); border-radius: 2px; vertical-align: middle; }}
.note {{ font-size: 0.85em; color: #777; margin-top: 1em; }}
</style>
</head>
<body>
<h1>Feature {feature_id}</h1>
<div class="meta">
model: <code>{model_id}</code> &middot; hook: <code>{hook_site}</code> &middot; layer: <code>{layer}</code> &middot; variant: <code>{variant}</code> &middot; d_sae: {d_sae}
</div>
<table>
<thead>
<tr><th>#</th><th>activation</th><th>token / context</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<div class="note">decoder ‖W_dec[f]‖ = {wdec_norm:.3f} &middot; sample size = {sample_size}</div>
</body>
</html>
"""


def _row(rank: int, act: float, max_act: float, context: str) -> str:
    bar_w = int(150 * (act / max_act)) if max_act > 0 else 0
    return (
        f"<tr><td>{rank}</td>"
        f"<td class='act'><span class='bar' style='width:{bar_w}px'></span> {act:.3f}</td>"
        f"<td><span class='tok'>{context}</span></td></tr>"
    )


@dataclass
class HTMLOutput:
    """Lightweight wrapper around an HTML string with `.save()` and `_repr_html_`."""

    html: str
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)

    def _repr_html_(self) -> str:  # Jupyter rich-repr
        return self.html

    def __str__(self) -> str:
        return self.html


def feature_explorer(
    sae: Any,
    feature_id: int,
    activations: Any | None = None,
    contexts: list[str] | None = None,
    top_k: int = 10,
) -> HTMLOutput:
    """Render the top-``k`` activating contexts for ``feature_id``.

    Parameters
    ----------
    sae : trained SAE
    feature_id : feature index
    activations : optional (N, d_in) tensor of cached activations. If None,
        we cannot compute top activations and produce a header-only page.
    contexts : optional list of length N of human-readable context strings
        (one per row of ``activations``). If shorter than N, we pad.
    top_k : how many top-activating rows to render.
    """
    import torch

    sample_size = 0
    rows_html = ""
    max_act = 0.0
    wdec_norm = float(sae.W_dec[feature_id].norm().item())

    if activations is not None:
        x = activations if isinstance(activations, torch.Tensor) else torch.as_tensor(activations)
        x = x.to(sae.W_enc.dtype)
        z = sae.encode(x)
        col = z[:, feature_id]
        sample_size = int(col.shape[0])
        k = min(top_k, sample_size)
        if k > 0:
            top_vals, top_idx = col.topk(k)
            max_act = float(top_vals[0].item())
            ctx_list = contexts or []
            for rank in range(k):
                act = float(top_vals[rank].item())
                idx = int(top_idx[rank].item())
                context = ctx_list[idx] if idx < len(ctx_list) else f"[sample idx {idx}]"
                rows_html += _row(rank + 1, act, max(max_act, 1e-9), context)

    html = _HTML_TEMPLATE.format(
        feature_id=feature_id,
        model_id=getattr(sae, "model_id", "") or "?",
        hook_site=getattr(sae, "hook_site", "?"),
        layer=getattr(sae, "layer", "?"),
        variant=getattr(sae, "variant", "?"),
        d_sae=getattr(sae, "d_sae", "?"),
        rows=rows_html
        or "<tr><td colspan='3' style='color:#999;'>No activations provided; pass activations=<i>tensor</i> for top-k preview.</td></tr>",
        wdec_norm=wdec_norm,
        sample_size=sample_size,
    )
    return HTMLOutput(
        html=html,
        meta={
            "feature_id": feature_id,
            "wdec_norm": wdec_norm,
            "sample_size": sample_size,
        },
    )
