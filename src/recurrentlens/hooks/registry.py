"""Forward-hook registration for Mamba blocks.

Phase 3. Implements the two hook sites listed in core.types.HOOK_SITES:

- ``out_proj_out``: residual-stream analog. Captured as the output of the
  Mamba block's ``out_proj`` (or the block's full output when ``out_proj``
  is not exposed). This is the recommended default for SAE training.

- ``ssm_h_t``: experimental recurrent-state capture. Resolved by attaching
  a forward hook to the SSM scan module (``mixer.ssm`` or the block itself
  as fallback). Recurrent state is not always exposed by HF Mamba; in that
  case the hook captures the SSM module's output as a proxy and emits a
  warning. Users wanting the true h_t should use nnsight-mode (opt-in,
  not enabled by default to keep the dependency surface small).

The HookManager is a small registry. It owns the live ``torch`` handles
and exposes ``.cache`` for downstream SAE training.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from recurrentlens.core.types import HookSite


@dataclass
class HookHandle:
    """Disposable handle returned when a hook is registered."""

    site: HookSite
    layer: int
    _torch_handle: Any
    _cache_ref: list[Any] = field(default_factory=list)

    def remove(self) -> None:
        if self._torch_handle is not None:
            self._torch_handle.remove()
            self._torch_handle = None

    @property
    def activations(self) -> list[Any]:
        """Return captured activations (one per forward pass)."""
        return list(self._cache_ref)

    def clear(self) -> None:
        self._cache_ref.clear()


_SSM_HT_WARNED: set[int] = set()


def resolve_target(layer_module: Any, site: HookSite) -> Any:
    """Pick the submodule that the forward hook attaches to."""
    if site == "out_proj_out":
        # HF Mamba has layer.mixer.out_proj. Legacy: layer.out_proj. Fallback: layer itself.
        for path in [("mixer", "out_proj"), ("out_proj",)]:
            obj = layer_module
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj
        return layer_module
    elif site == "ssm_h_t":
        # Closest accessible SSM submodule; true recurrent state requires nnsight.
        for path in [("mixer", "ssm"), ("mixer",)]:
            obj = layer_module
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                key = id(obj)
                if key not in _SSM_HT_WARNED:
                    _SSM_HT_WARNED.add(key)
                    warnings.warn(
                        "ssm_h_t hook attached to closest SSM submodule output, "
                        "which is a proxy for the recurrent state. For exact h_t, "
                        "use the nnsight backend (recurrentlens[nnsight], v0.1.x). "
                        "This warning is emitted once per layer module.",
                        UserWarning,
                        stacklevel=3,
                    )
                return obj
        return layer_module
    else:
        raise ValueError(f"unknown hook site: {site!r}")


# Back-compat alias for internal callers; will be removed in v0.2.
_resolve_target = resolve_target


def register_hook(
    model: Any,
    site: HookSite,
    layer: int,
    transform: Callable[[Any], Any] | None = None,
) -> HookHandle:
    """Attach a forward hook on ``model`` at the given site and layer.

    Parameters
    ----------
    model : RecurrentLensModelImpl (or any object exposing .get_layer(idx))
    site : "out_proj_out" | "ssm_h_t"
    layer : int
    transform : optional callable applied to the captured tensor before caching.
        Identity by default. Useful e.g. for ``.detach().cpu()`` to spill GPU.

    Returns
    -------
    HookHandle with ``.activations`` list and ``.remove()`` method.
    """
    target = resolve_target(model.get_layer(layer), site)
    cache: list[Any] = []

    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        if transform is not None:
            out = transform(out)
        cache.append(out)

    torch_handle = target.register_forward_hook(_hook)
    return HookHandle(site=site, layer=layer, _torch_handle=torch_handle, _cache_ref=cache)


class HookManager:
    """Manage multiple hooks at once and bulk-remove on context exit."""

    def __init__(self, model: Any):
        self.model = model
        self._handles: list[HookHandle] = []

    def add(
        self,
        site: HookSite,
        layer: int,
        transform: Callable[[Any], Any] | None = None,
    ) -> HookHandle:
        h = register_hook(self.model, site=site, layer=layer, transform=transform)
        self._handles.append(h)
        return h

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> HookManager:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove_all()
