"""Concrete RecurrentLensModel wrapping a HuggingFace Mamba checkpoint.

Phase 2 implementation. Provides:
- load_model_impl(model_id, device, dtype, revision) factory
- RecurrentLensModelImpl class with .forward(), .hook_sites(), .get_layer()

The recurrent-state-aware hook registration itself lives in
recurrentlens.hooks (Phase 3). This module is just the model wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recurrentlens.core.types import HOOK_SITES, HookSite


@dataclass
class _MambaConfigShim:
    """Minimal config extraction used by the wrapper.

    Real HF MambaConfig has many fields; we keep only what the wrapper needs.
    Pulled out as a dataclass so unit tests can construct it without HF.
    """

    n_layers: int
    d_model: int
    vocab_size: int
    model_type: str = "mamba"

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> _MambaConfigShim:
        # HF Mamba: hidden_size + num_hidden_layers; legacy raw mamba: d_model + n_layer.
        n_layers = getattr(hf_config, "num_hidden_layers", None) or getattr(hf_config, "n_layer", 0)
        d_model = getattr(hf_config, "hidden_size", None) or getattr(hf_config, "d_model", 0)
        vocab_size = getattr(hf_config, "vocab_size", 0)
        model_type = getattr(hf_config, "model_type", "mamba")
        return cls(
            n_layers=int(n_layers),
            d_model=int(d_model),
            vocab_size=int(vocab_size),
            model_type=model_type,
        )


class RecurrentLensModelImpl:
    """Wrapper around a HuggingFace Mamba/SSM model.

    Attributes
    ----------
    model_id : str
    n_layers : int
    d_model : int
    device : str
    dtype : Any (torch dtype if torch is imported; opaque otherwise)
    hf_model : Any (the underlying HF model)
    tokenizer : Any (HF tokenizer)
    """

    def __init__(
        self,
        hf_model: Any,
        tokenizer: Any,
        model_id: str,
        device: str,
        dtype: Any,
        config: _MambaConfigShim,
    ):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._config = config
        self.n_layers = config.n_layers
        self.d_model = config.d_model

    def __repr__(self) -> str:
        return (
            f"RecurrentLensModel(model_id={self.model_id!r}, "
            f"n_layers={self.n_layers}, d_model={self.d_model}, "
            f"device={self.device!r}, dtype={self.dtype})"
        )

    def hook_sites(self) -> list[HookSite]:
        """Return supported hook site names."""
        return list(HOOK_SITES)

    def get_layer(self, idx: int) -> Any:
        """Return the Mamba block at index ``idx``.

        Supports both HF AutoModelForCausalLM Mamba layout
        (model.backbone.layers) and the legacy state-spaces/mamba layout
        (model.layers / model.model.layers).
        """
        if not 0 <= idx < self.n_layers:
            raise IndexError(f"layer {idx} out of range [0, {self.n_layers})")
        m = self.hf_model
        for path in [
            ("backbone", "layers"),
            ("model", "layers"),
            ("layers",),
        ]:
            obj: Any = m
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj[idx]
        raise AttributeError(f"could not locate layer container on {type(self.hf_model).__name__}")

    def forward(self, input_ids: Any) -> Any:
        return self.hf_model(input_ids)

    def __call__(self, input_ids: Any) -> Any:
        return self.forward(input_ids)


def _resolve_device_dtype(device: str, dtype: Any) -> tuple[str, Any]:
    """Resolve 'auto' device and default dtype with torch (lazy-imported)."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


def load_model_impl(
    model_id: str,
    device: str = "auto",
    dtype: Any = None,
    revision: str | None = None,
) -> RecurrentLensModelImpl:
    """Load a Mamba/SSM model from HuggingFace and wrap it.

    Performs network I/O; use the integration test marker to gate calls in CI.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = _resolve_device_dtype(device, dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, revision=revision)
    hf_model = hf_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    config = _MambaConfigShim.from_hf_config(hf_model.config)
    return RecurrentLensModelImpl(
        hf_model=hf_model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=device,
        dtype=dtype,
        config=config,
    )
