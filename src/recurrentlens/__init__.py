"""recurrentlens: mechanistic interpretability for State-Space Models."""

from recurrentlens._version import __version__
from recurrentlens.core.model import RecurrentLensModelImpl, load_model_impl
from recurrentlens.core.protocols import SAE, RecurrentLensModel
from recurrentlens.core.types import (
    CACHE_BACKENDS,
    DEFAULT_HOOK_SITE,
    DEFAULT_SAE_VARIANT,
    HOOK_SITES,
    SAE_VARIANTS,
    CacheBackend,
    HookSite,
    SAEVariant,
)

__all__ = [
    "__version__",
    "SAE",
    "RecurrentLensModel",
    "RecurrentLensModelImpl",
    "HookSite",
    "SAEVariant",
    "CacheBackend",
    "HOOK_SITES",
    "SAE_VARIANTS",
    "CACHE_BACKENDS",
    "DEFAULT_HOOK_SITE",
    "DEFAULT_SAE_VARIANT",
    "load_model",
    "train_sae",
    "ablate",
    "steer",
    "viz",
    "hub",
    "bench",
]


def load_model(
    model_id: str,
    device: str = "auto",
    dtype=None,
    revision: str | None = None,
) -> RecurrentLensModelImpl:
    """Load a Mamba/SSM model and wrap it as a RecurrentLensModel.

    Performs network I/O when called.
    """
    return load_model_impl(model_id=model_id, device=device, dtype=dtype, revision=revision)


def train_sae(
    model,
    hook_site: HookSite = "out_proj_out",
    layer: int = 0,
    variant: SAEVariant = "topk",
    d_sae: int = 16384,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    n_tokens: int = 100_000_000,
    cache_backend: CacheBackend = "mmap",
    save_to: str | None = None,
    **kwargs,
):
    """Train a sparse autoencoder on a hooked activation site.

    Implemented in Phase 5 (training loop). Currently raises NotImplementedError.
    """
    raise NotImplementedError("train_sae: Phase 5 training loop not yet wired")


def ablate(model, sae, feature_id, strength: float = 0.0):
    """Zero-ablate (or scale-ablate) an SAE feature during the model's forward pass."""
    raise NotImplementedError("ablate: Phase 7 not yet implemented")


def steer(model, sae, feature_id, vector_scale: float = 1.0):
    """Context-managed feature steering."""
    raise NotImplementedError("steer: Phase 7 not yet implemented")


class _DeferredSubmodule:
    def __init__(self, name: str, phase: str):
        self._name = name
        self._phase = phase

    def __getattr__(self, attr: str):
        raise NotImplementedError(
            f"recurrentlens.{self._name}.{attr}: {self._phase} not yet implemented"
        )


viz = _DeferredSubmodule("viz", "Phase 8")
hub = _DeferredSubmodule("hub", "Phase 8")
bench = _DeferredSubmodule("bench", "Phase 9")
