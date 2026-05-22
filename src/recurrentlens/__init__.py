"""recurrentlens: mechanistic interpretability for State-Space Models."""

from recurrentlens import bench, hub, viz
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
from recurrentlens.features.ablate import ablate
from recurrentlens.features.steer import steer

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
    """Load a Mamba/SSM model and wrap it as a RecurrentLensModel."""
    return load_model_impl(model_id=model_id, device=device, dtype=dtype, revision=revision)


def train_sae(
    model,
    hook_site: HookSite = "out_proj_out",
    layer: int = 0,
    variant: SAEVariant = "topk",
    d_sae: int = 16384,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    n_tokens: int = 5_000_000,
    cache_backend: CacheBackend = "mmap",
    save_to: str | None = None,
    n_steps: int = 1000,
    **kwargs,
):
    """End-to-end SAE training. See ``recurrentlens.sae.train.train_sae_full`` for details."""
    from recurrentlens.sae.train import train_sae_full

    return train_sae_full(
        model=model,
        hook_site=hook_site,
        layer=layer,
        variant=variant,
        d_sae=d_sae,
        dataset=dataset,
        n_tokens=n_tokens,
        cache_backend=cache_backend,
        save_to=save_to,
        n_steps=n_steps,
        **kwargs,
    )
