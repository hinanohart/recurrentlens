"""Zero/scale-ablation of an SAE feature in the live forward pass.

Phase 7. ``ablate`` removes (or scales) a single feature's contribution to
the residual stream. The contribution is computed by SAE-encoding the
captured activation, masking the target feature, and re-decoding, then
substituting the masked reconstruction in place of the original.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


@contextmanager
def ablate(model: Any, sae: Any, feature_id: int | list[int], strength: float = 0.0):
    """Scale-ablate one or more SAE features.

    ``strength=0.0`` zeros out the feature; ``strength=0.5`` halves it; etc.
    Other features are left untouched.
    """
    from recurrentlens.hooks.registry import _resolve_target

    layer_module = model.get_layer(sae.layer)
    target = _resolve_target(layer_module, sae.hook_site)
    feats = [feature_id] if isinstance(feature_id, int) else list(feature_id)

    def _ablate_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        primary = output[0] if isinstance(output, tuple) else output
        d_in = primary.shape[-1]
        flat = primary.reshape(-1, d_in)
        with_dtype = flat.to(sae.W_enc.dtype)
        z = sae.encode(with_dtype)
        scale = z.new_ones(z.shape[-1])
        for f in feats:
            scale[f] = strength
        z_masked = z * scale
        x_hat_masked = sae.decode(z_masked).to(primary.dtype).reshape(primary.shape)
        x_hat = sae.decode(z).to(primary.dtype).reshape(primary.shape)
        modified = primary + (x_hat_masked - x_hat)
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = target.register_forward_hook(_ablate_hook)
    try:
        yield
    finally:
        handle.remove()
