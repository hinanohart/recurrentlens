"""recurrentlens: mechanistic interpretability for State-Space Models."""

from recurrentlens._version import __version__
from recurrentlens.core.protocols import SAE, RecurrentLensModel

__all__ = [
    "__version__",
    "SAE",
    "RecurrentLensModel",
    "load_model",
    "train_sae",
    "ablate",
    "steer",
    "viz",
    "hub",
    "bench",
]


def load_model(model_id: str, device: str = "auto", dtype=None, revision: str | None = None):
    """Load a Mamba/SSM model and wrap it as a RecurrentLensModel.

    Implemented in Phase 2. Currently raises NotImplementedError.
    """
    raise NotImplementedError("load_model: Phase 2 not yet implemented; tracking #2")


def train_sae(
    model,
    hook_site: str = "out_proj_out",
    layer: int = 0,
    variant: str = "topk",
    d_sae: int = 16384,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    n_tokens: int = 100_000_000,
    cache_backend: str = "mmap",
    save_to: str | None = None,
):
    """Train a sparse autoencoder on a hooked activation site.

    Implemented in Phases 4-5. Currently raises NotImplementedError.
    """
    raise NotImplementedError("train_sae: Phases 4-5 not yet implemented")


def ablate(model, sae, feature_id, strength: float = 0.0):
    """Zero-ablate (or scale-ablate) an SAE feature during the model's forward pass.

    Implemented in Phase 7. Currently raises NotImplementedError.
    """
    raise NotImplementedError("ablate: Phase 7 not yet implemented")


def steer(model, sae, feature_id, vector_scale: float = 1.0):
    """Context-managed feature steering.

    Implemented in Phase 7. Currently raises NotImplementedError.
    """
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
