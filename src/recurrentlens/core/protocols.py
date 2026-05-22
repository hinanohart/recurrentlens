"""Protocol definitions for the recurrentlens public API.

These are intentionally minimal in Phase 1 and will be filled in during Phase 2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RecurrentLensModel(Protocol):
    """A model that exposes the recurrent-state-aware hook API.

    Phase 2 will provide a concrete implementation wrapping HuggingFace Mamba
    checkpoints; this protocol pins the contract.
    """

    model_id: str
    n_layers: int
    d_model: int

    def forward(self, input_ids): ...

    def hook_sites(self) -> list[str]:
        """Return supported hook site names (e.g. 'out_proj_out', 'ssm_h_t')."""
        ...


@runtime_checkable
class SAE(Protocol):
    """Sparse autoencoder protocol.

    Phase 4 will provide vanilla/TopK/JumpReLU implementations.
    """

    d_in: int
    d_sae: int
    variant: str
    hook_site: str
    layer: int

    def encode(self, x): ...

    def decode(self, z): ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> SAE: ...
