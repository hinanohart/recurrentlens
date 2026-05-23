"""Feature steering context manager.

Phase 7. ``steer`` adds the SAE's decoder direction for ``feature_id`` to the
residual stream during the model's forward pass, scaled by ``vector_scale``.

Usage::

    with rl.steer(model, sae, feature_id=42, vector_scale=2.0):
        output = model.forward(input_ids)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


@contextmanager
def steer(model: Any, sae: Any, feature_id: int, vector_scale: float = 1.0):
    """Inject ``vector_scale * sae.W_dec[feature_id]`` at the SAE's layer/site."""
    import torch

    from recurrentlens.hooks.registry import resolve_target

    layer_module = model.get_layer(sae.layer)
    target = resolve_target(layer_module, sae.hook_site)

    direction = sae.W_dec[feature_id].detach()  # (d_in,)

    def _add_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            primary = output[0]
            modified = primary + (vector_scale * direction.to(primary.dtype).to(primary.device))
            return (modified, *output[1:])
        return output + (vector_scale * direction.to(output.dtype).to(output.device))

    handle = target.register_forward_hook(_add_hook)
    try:
        yield
    finally:
        handle.remove()
        del torch  # quiet unused import lint
